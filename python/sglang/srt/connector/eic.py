# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple

import torch
import ctypes
import os
import yaml
import time
import logging
import numpy as np
import json
import signal
import eic
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
import concurrent.futures

from sglang.srt.connector import BaseKVConnector

logger = logging.getLogger(__name__)

dtype_to_int = {
    torch.bool: 0,
    torch.float8_e4m3fn: 1,
    torch.float8_e5m2: 2,
    torch.float16: 3,
    torch.bfloat16: 4,
    torch.float32: 5,
    torch.float64: 6,
    torch.int8: 7,
    torch.int16: 8,
    torch.int32: 9,
    torch.int64: 10,
    torch.uint8: 11,
    torch.uint16: 12,
    torch.uint32: 13,
    torch.uint64: 14,
}

int_to_dtype = {
    0: torch.bool,
    1: torch.float8_e4m3fn,
    2: torch.float8_e5m2,
    3: torch.float16,
    4: torch.bfloat16,
    5: torch.float32,
    6: torch.float64,
    7: torch.int8,
    8: torch.int16,
    9: torch.int32,
    10: torch.int64,
    11: torch.uint8,
    12: torch.uint16,
    13: torch.uint32,
    14: torch.uint64,
}


def get_int_from_dtype(dtype: torch.dtype) -> int:
    return dtype_to_int[dtype]


def get_dtype_from_int(dtype: int) -> torch.dtype:
    return int_to_dtype[dtype]

def generate_tensor_metadata(tensor: torch.Tensor) -> dict[str, any]:
    meta : dict[str, any] = {}
    # dtype, shape
    meta["dtype"] = get_int_from_dtype(tensor.dtype)
    meta["shape"] = tuple(tensor.shape)
    return meta

def get_tensor_from_metadata(meta: dict[str, any]) -> torch.Tensor:
    dtype = get_dtype_from_int(meta["dtype"])
    return torch.empty(meta["shape"], dtype=dtype)


def _make_dir(path: str):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        logger.info(f"create dir '{path}' success")
    except OSError as e:
        logger.info(f"create dir '{path}' error {e}")
        exit(1)


def _get_config_key(model_path: str) -> str:
    return os.path.join(model_path, "config")


def _get_config_file_key(model_path: str, filename: str) -> str:
    return os.path.join(model_path, filename)


def pull_files_from_eic(model_path: str, local_dir: str):
    logger.info(
        f"pull_files_from_eic... local_dir = {local_dir} PID = {os.getpid()}")

    pull_config_file = os.path.join(local_dir, EICConnector.PULL_CONFIG_FILE)
    if os.path.exists(pull_config_file):
        logger.info(
            f"config file exists, no need to pull again, local_dir = {local_dir} PID = {os.getpid()}")
        return

    import multiprocessing
    p = multiprocessing.Process(
        target=do_pull_files_from_eic, args=(model_path, local_dir))
    p.start()
    p.join()


def do_pull_files_from_eic(model_path: str, local_dir: str):
    client = EICConnector.instance(model_path)
    client.set_local_dir(local_dir)
    client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
    EICConnector.release_instance()


class EICConnector(BaseKVConnector):
    PULL_CONFIG_FILE = ".eic_model_config"

    def __init__(self, url: str):
        super().__init__(url)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal.SIG_DFL)

        config_file = '/sgl-workspace/config/remote-eic.yaml'
        with open(config_file, "r") as fin:
            config = yaml.safe_load(fin)

        remote_url = config.get("remote_url", None)
        if remote_url is None:
            AssertionError("remote_url is None")

        endpoint = remote_url[len('eic://'):]
        eic_instance_id = config.get("eic_instance_id", None)
        eic_thread_num = config.get("eic_thread_num", 6)
        eic_log_dir = config.get("eic_log_dir", None)
        eic_log_level = config.get("eic_log_level", 1)
        # tcp: 0, rdma: 2
        eic_trans_type = config.get("eic_trans_type", 2)
        eic_flag_file = config.get("eic_flag_file", None)
        eic_s3_ak = config.get("eic_s3_ak", "")
        eic_s3_sk = config.get("eic_s3_sk", "")
        eic_s3_endpoint = config.get("eic_s3_endpoint", "")
        eic_s3_region = config.get("eic_s3_region", "")

        # self.bucket_name = "eic-models-bucket"
        self.bucket_name = config.get("eic_s3_bucket_name", "")
        self.eic_model_namespace = config.get("eic_model_namespace", "")

        logger.info(
            f"eic init start, pid: {os.getpid()}, "
            f"endpoint: {endpoint}, "
            f"eic_instance_id: {eic_instance_id}, "
            f"eic_thread_num: {eic_thread_num}, "
            f"eic_log_dir: {eic_log_dir}, "
            f"eic_log_level: {eic_log_level}, "
            f"eic_trans_type: {eic_trans_type}, "
            f"eic_flag_file: {eic_flag_file}, "
            f"eic_s3_ak : {eic_s3_ak}, "
            f"eic_s3_sk: {eic_s3_sk}, "
            f"eic_s3_endpoint: {eic_s3_endpoint}, "
            f"eic_s3_region: {eic_s3_region}, "
            f"eic_s3_bucket_name: {self.bucket_name}, "
            f"eic_model_namespace: {self.eic_model_namespace}")
        _make_dir(eic_log_dir)

        self.thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get("eic_thread_pool_size", 32))
        self.connection = eic.Client()
        init_option = eic.InitOption()
        init_option.log_dir = eic_log_dir
        init_option.log_level = eic.LogLevel(eic_log_level)
        init_option.transport_type = eic.TransportType(eic_trans_type)
        if eic_flag_file:
            init_option.flag_file = eic_flag_file
        ret = self.connection.init(eic_instance_id, endpoint, init_option)
        if ret != 0:
            logger.error(f"fail to init eic client, ret: {ret}")
            exit(1)

        logger.info(f"eic init success")

        s3_client = boto3.client('s3', region_name=eic_s3_region,
                          endpoint_url=eic_s3_endpoint,
                          aws_access_key_id=eic_s3_ak,
                          aws_secret_access_key=eic_s3_sk,
                          config=Config(s3={'addressing_style': 'virtual'}))

        self.object_storage_client = s3_client
        logger.info(f"object client init success")

    @classmethod
    def instance(cls, *args, **kwargs):
        if not hasattr(EICConnector, "_instance"):
            EICConnector._instance = EICConnector(*args, **kwargs)
        return EICConnector._instance

    @classmethod
    def release_instance(cls):
        if not hasattr(EICConnector, "_instance"):
            return
        del EICConnector._instance

    def get(self, key: str) -> Optional[torch.Tensor]:
        tensors = self.mget([key])
        return tensors[0] if tensors else None

    def memcpy_tensor(self, dst_tensor: torch.Tensor, src_bytes: bytes):
        if not dst_tensor.device.type == 'cpu':
            raise ValueError("memcpy_tensor only supports CPU tensors")
        dst_ptr = dst_tensor.data_ptr()
        dst_size = dst_tensor.element_size() * dst_tensor.numel()
        if len(src_bytes) != dst_size:
            raise ValueError(f"Size mismatch: dst tensor size = {dst_size}, src bytes = {len(src_bytes)}")
        src_buffer = (ctypes.c_char * len(src_bytes)).from_buffer_copy(src_bytes)
        ctypes.memmove(dst_ptr, src_buffer, dst_size)

    def mget(self, keys: List[str], tensors: Optional[List[torch.Tensor]] = None) -> Optional[List[torch.Tensor]]:
        logger.debug(f"eic get {keys}")
        get_data_start_time = time.perf_counter()
        data_keys = eic.StringVector()
        data_keys.extend(keys)
        if tensors is not None:
            data_vals = eic.IOBuffers()
            for tensor in tensors:
                data_vals.append(tensor.data_ptr(), tensor.element_size()
                                 * tensor.numel(), False)
        else:
            data_vals = None
        get_option = eic.GetOption()
        get_option.ns = self.eic_model_namespace
        status_code, data_vals, get_outcome = self.connection.mget(
            data_keys, get_option, data_vals)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mget {keys} failed, status_code {status_code} PID = {os.getpid()}")
            for i, err_code in enumerate(get_outcome.status_codes):
                if err_code != eic.StatusCode.SUCCESS:
                    logger.error(
                        f"eic mget {keys[i]} failed, err_code {err_code} PID = {os.getpid()}")
        get_data_end_time = time.perf_counter()
        get_data_execution_time = (
            get_data_end_time - get_data_start_time) * 1e6
        logger.debug(f"eic get {keys} data cost %.2f ms",
                     get_data_execution_time * 1e3)

        result_tensors = [None] * len(keys)
        object_futures = []

        for i, key in enumerate(keys):
            success = (
                status_code == eic.StatusCode.SUCCESS and
                get_outcome.status_codes[i] == eic.StatusCode.SUCCESS
            )

            if success:
                if tensors is None:
                    byte_data = data_vals[i].encode()
                    tensor = torch.from_numpy(np.frombuffer(byte_data, dtype=np.int8).copy())
                    result_tensors[i] = tensor
            else:
                logger.debug(f"eic mget failed for key: {key}, fallback to object storage")
                object_futures.append( (i, self._get_from_object_storage_async(key)) )

        for i, future in object_futures:
            content, tensor = future.result()
            if content is None:
                logger.error(f"Failed to get tensor from both EIC and object storage for key: {keys[i]}")
                return None
            else:
                if tensors is None:
                    result_tensors[i] = tensor
                else:
                    self.memcpy_tensor(tensors[i], content)
                    result_tensors[i] = tensor

        if tensors is None:
            return result_tensors
        else:
            return tensors

    def getstr(self, key: str) -> Optional[str]:
        val = self.connection.get(key)
        if val is None:
            logger.error("Key %s not found", key)
            return None

        return val.decode("utf-8")

    def mset(self, input_keys: List[str], tensors: List[torch.Tensor]) -> None:
        logger.debug(f"eic set {input_keys}")
        keys = eic.StringVector()
        vals = eic.IOBuffers()
        cpu_tensor_refs: List[torch.Tensor] = []
        keys.extend(input_keys)
        for tensor in tensors:
            cpu_tensor = tensor.cpu()
            cpu_tensor_refs.append(cpu_tensor)
            vals.append(cpu_tensor.data_ptr(), cpu_tensor.element_size()
                        * cpu_tensor.numel(), False)
        set_option = eic.SetOption()
        set_option.ns = self.eic_model_namespace
        set_option.ttl_second = -1
        status_code, set_outcome = self.connection.mset(keys, vals, set_option)
        if status_code != eic.StatusCode.SUCCESS:
            logger.error(f"eic mset {keys} failed, status_code {status_code}")
            for i, err_code in enumerate(set_outcome.status_codes):
                if err_code != eic.StatusCode.SUCCESS:
                    logger.error(
                        f"eic mset {keys[i]} failed, err_code {err_code}")
            raise RuntimeError(f"eic mset failed, code: {status_code}")
        for i, key in enumerate(input_keys):
            self._set_to_object_storage(key, tensors[i])

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        tensor = tensor.cpu().contiguous()
        num_bytes = tensor.element_size() * tensor.numel()
        data_ptr = tensor.data_ptr()
        buf_type = ctypes.c_char * num_bytes
        byte_buf = buf_type.from_address(data_ptr)
        return bytes(byte_buf)

    def _set_to_object_storage(self, key: str, tensor: torch.Tensor) -> None:
        """Upload tensor to S3"""
        byte_data = self._tensor_to_bytes(tensor)
        try:
            self.object_storage_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=byte_data
            )
            logger.info(f"S3 put_object success, bucket={self.bucket_name}, key={key}")
        except ClientError as e:
            logger.error(f"S3 put_object failed, bucket={self.bucket_name}, key={key}, error={e}")
        except Exception as e:
            logger.error(f"S3 put_object unexpected error, bucket={self.bucket_name}, key={key}, error={e}")

    def _get_from_object_storage(self, key: str) -> (bytes, torch.Tensor):
        """Download tensor from S3"""
        logger.debug(f"S3 get_object start, bucket={self.bucket_name}, key={key}")
        try:
            response = self.object_storage_client.get_object(Bucket=self.bucket_name, Key=key)
            content = response["Body"].read()
            np_array = np.frombuffer(content, dtype=np.int8).copy()
            tensor = torch.from_numpy(np_array)
            logger.debug(f"S3 get_object success, bucket={self.bucket_name}, key={key}")
            return content, tensor
        except ClientError as e:
            logger.error(f"S3 get_object failed, bucket={self.bucket_name}, key={key}, error={e}")
            return None, None
        except Exception as e:
            logger.error(f"S3 get_object unexpected error, bucket={self.bucket_name}, key={key}, error={e}")
            return None, None

    def _get_from_object_storage_async(self, key: str) -> concurrent.futures.Future:
        def _object_storage_get_task():
            logger.debug(f"S3 get_object start, bucket={self.bucket_name}, key={key}")
            try:
                response = self.object_storage_client.get_object(Bucket=self.bucket_name, Key=key)
                content = response["Body"].read()
                np_array = np.frombuffer(content, dtype=np.int8).copy()
                tensor = torch.from_numpy(np_array)
                logger.debug(f"S3 get_object success, bucket={self.bucket_name}, key={key}")
                return content, tensor
            except ClientError as e:
                logger.error(f"S3 get_object failed, bucket={self.bucket_name}, key={key}, error={e}")
                return None, None
            except Exception as e:
                logger.error(f"S3 get_object unexpected error, bucket={self.bucket_name}, key={key}, error={e}")
                return None, None

        return self.thread_pool.submit(_object_storage_get_task)

    def set(self, key: str, tensor: torch.Tensor) -> None:
        assert tensor is not None
        self.mset([key], [tensor])

    def setstr(self, key: str, obj: str) -> None:
        self.connection.set(key, obj)

    def pull_files(
        self,
        allow_pattern: Optional[List[str]] = None,
        ignore_pattern: Optional[List[str]] = None,
    ) -> None:
        """
        download model config from eic
        """
        model_name = self.url.removeprefix("eic://")
        config_key = _get_config_key(model_name)
        logger.debug(f"start download model config from eic, key: {config_key}, path {self.local_dir}")
        keys = [config_key]
        tensors = self.mget(keys)
        if tensors is None:
            logger.error(
                f"download model config from eic failed, key: {config_key}")
            return False
        bytes_data = tensors[0].numpy().tobytes()
        config_json = json.loads(bytes_data.decode())
        file_name_list = config_json["file_name_list"]
        data_keys: List[str] = []
        data_tensors: List[torch.Tensor] = []
        for i, filename in enumerate(file_name_list):
            data_keys.append(_get_config_file_key(model_name, filename))
        batch_size = 20
        for i in range(0, len(data_keys), batch_size):
            batch_keys = data_keys[i:i + batch_size]
            batch_tensors = self.mget(batch_keys)
            data_tensors.extend(batch_tensors)
        logger.debug(f"download finished, model config from eic, key: {config_key}, path {self.local_dir}")
        _make_dir(self.local_dir)
        for i, tensor in enumerate(data_tensors):
            filename = file_name_list[i]
            local_path_file = os.path.join(self.local_dir, filename)
            if os.path.exists(local_path_file) and os.path.getsize(local_path_file) == tensor.numel() * tensor.element_size():
                logger.info(
                    f"model config file {local_path_file} already exists, skip")
                continue
            with open(local_path_file, "wb") as fout:
                fout.write(tensor.numpy().tobytes())
                logger.info(
                    f"download model config from eic success, file: {local_path_file} PID = {os.getpid()}")

        config_file = os.path.join(
            self.local_dir, EICConnector.PULL_CONFIG_FILE)
        with open(config_file, "wb") as fout:
            fout.write(bytes_data)

    def upload_model_config(self, model_name: str, file_list: List[str]):
        """
        upload model config to eic
        """
        file_num = len(file_list)
        file_name_total_length = 0
        base_name_list: List[str] = []
        for i, filename in enumerate(file_list):
            base_name = os.path.basename(filename)
            base_name_list.append(base_name)
            file_name_total_length += len(base_name)
        # json format: { file_name_list[]}
        config_json = {
            "file_num": file_num,
            "file_name_list": base_name_list
        }
        config_json_bytes = json.dumps(config_json).encode()
        config_buffer = torch.from_numpy(np.frombuffer(
            config_json_bytes, dtype=np.int8).copy())
        logger.info(f"start upload model config to eic, json: {config_json}")
        keys: List[str] = [_get_config_key(model_name)]
        tensors: List[torch.Tensor] = [config_buffer]
        # read file and make tensor
        for i, filename in enumerate(file_list):
            with open(filename, "rb") as fin:
                buffer = fin.read()
            tensor = torch.from_numpy(np.frombuffer(buffer, dtype=np.int8).copy())
            keys.append(_get_config_file_key(model_name, base_name_list[i]))
            tensors.append(tensor)
        batch_size = 20
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            batch_tensors = tensors[i:i + batch_size]
            self.mset(batch_keys, batch_tensors)
        logger.info(f"upload model config to eic done, file num: {file_num}")

    def weight_iterator(
        self, rank: int = 0
    ) -> Generator[Tuple[str, bytes], None, None]:
        keys = self.list(f"{self.model_name}/keys/rank_{rank}/")
        for key in keys:
            val = self.get(key)
            key = key.removeprefix(f"{self.model_name}/keys/rank_{rank}/")
            yield key, val

    def close(self) -> None:
        if self.connection:
            self.connection = None
        # 新增：关闭线程池（等待所有任务完成）
        if hasattr(self, 'thread_pool') and self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        self.closed = True
        super().close()

    def allocate_zero_copy_buffer(self, length):
        return self.connection.allocate_managed_buffer(length)

    def release_zero_copy_buffer(self, ptr, length):
        return self.connection.free_managed_buffer(ptr, length)

    def list(self, prefix: str) -> List[str]:
        pass
