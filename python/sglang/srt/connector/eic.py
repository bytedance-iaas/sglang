# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Generator, List, Optional, Tuple, Dict, Any

import torch
import ctypes
import os
import yaml
import time
import logging
import numpy as np
import json
import signal
import math
import ctypes
import threading
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


def pull_files_from_eic(model_path: str):
    model_name = model_path.removeprefix("eic://")
    local_dir = "/tmp/" + "_".join([model_name, str(os.getpid())])
    os.makedirs(local_dir, exist_ok=True)

    pull_config_file = os.path.join(local_dir, EICConnector.PULL_CONFIG_FILE)
    if os.path.exists(pull_config_file):
        logger.info(
            f"config file exists, no need to pull again, local_dir = {local_dir} PID = {os.getpid()}")
        return local_dir

    import multiprocessing
    p = multiprocessing.Process(
        target=do_pull_files_from_eic, args=(model_path, local_dir))
    p.start()
    p.join()
    return local_dir


def do_pull_files_from_eic(model_path: str, local_dir: str):
    client = EICConnector.instance(model_path)
    client.set_local_dir(local_dir)
    client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
    EICConnector.release_instance()


class EICConnector(BaseKVConnector):
    PULL_CONFIG_FILE = ".eic_model_config"

    MODEL_BATCH_SET_SIZE = 20 * 1024 * 1024
    MODEL_DATA_SPLIT_SIZE = 20 * 1024 * 1024
    MODEL_META_APPROXIMATE_SIZE = 10 * 1024 * 1024
    MODEL_META_PREFIX_MAX_LENGTH = 16
    MODEL_META_PREFIX = "._meta"
    ENABLE_ZERO_COPY = True
    ENABLE_THREADING = False
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
        self.loaded_kv: dict[str, Any] = {}
        self.perf_statistics: dict[str, Any] = {}
        self.thread_num = 4
        self.zero_copy_ptrs = []
        self.split_tensors = []

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

    def _write_huge_kv(
        self,
        model_name: str,
        key: str,
        tensor: torch.Tensor,
        data_split_size: int,
    ) -> None:
        # split tensor
        tensor_size = tensor.nelement() * tensor.element_size()
        split_num = math.ceil(tensor_size / data_split_size)
        for i in range(split_num):
            start = i * data_split_size
            end = min((i + 1) * data_split_size, tensor_size)
            split_tensor = tensor.view(
                tensor.nelement()).view(torch.int8)[start: end]
            split_key = os.path.join(model_name, key, str(i))
            self.mset([split_key], [split_tensor])

    def save_model(self, model_prefix, model_name, local_model_path, rank, state_dict):
        start_time = time.perf_counter()
        # root: ( k = model_prefix , v = { kv_num, meta_number, data_split_size, meta_prefix_name, kv_num_per_meta[] } )
        root_json = {
            "kv_num": 0,
            "meta_num": 0,
            "data_split_size": self.MODEL_DATA_SPLIT_SIZE,
            "meta_prefix_name": self.MODEL_META_PREFIX,
            "kv_num_per_meta": [],
        }

        # meta:( k = model_prefix/meta_prefix_name/{index} , v = { kv_num, key[], tensor_meta[], value_length[] })
        meta_size = 0
        meta_json = {
            "kv_num": 0,
            "key": [],
            "tensor_meta": [],
            "value_length": [],
        }
        # data: ( k = key, v = tensor )
        # set tensors
        keys: List[str] = []
        tensors: List[torch.Tensor] = []
        batch_size = 0
        for key, tensor in state_dict.items():
            should_write_data = False
            # update meta
            root_json["kv_num"] += 1
            meta_json["kv_num"] += 1
            tensor_size = tensor.nelement() * tensor.element_size()
            meta_json["value_length"].append(tensor_size)
            tensor_meta = generate_tensor_metadata(tensor)
            meta_json["tensor_meta"].append(tensor_meta)
            meta_json["key"].append(key)
            # logger.info(f"model_prefix {model_prefix} save key {key} tensor_size {tensor_size} tensor_meta {tensor_meta}")
            meta_size += len(key)
            if tensor_size > self.MODEL_DATA_SPLIT_SIZE:
                should_write_data = True
                self._write_huge_kv(
                    model_prefix,
                    key,
                    tensor,
                    self.MODEL_DATA_SPLIT_SIZE,
                )
            else:
                # append to write batch
                keys.append(os.path.join(model_prefix, key))
                tensors.append(tensor)
                batch_size = batch_size + tensor_size
                if batch_size >= self.MODEL_BATCH_SET_SIZE:
                    should_write_data = True
                    self.mset(keys, tensors)
                    keys.clear()
                    tensors.clear()
                    batch_size = 0
            # write meta if needed
            if should_write_data and meta_size >= self.MODEL_META_APPROXIMATE_SIZE:
                meta_name = os.path.join(
                    model_prefix, self.MODEL_META_PREFIX, str(root_json["meta_num"]))
                meta_bytes = json.dumps(meta_json).encode()
                meta_buffer = torch.from_numpy(np.frombuffer(
                    meta_bytes, dtype=np.int8).copy())
                self.mset([meta_name], [meta_buffer])
                # update root
                root_json["kv_num_per_meta"].append(meta_json["kv_num"])
                root_json["meta_num"] += 1
                # reset counters
                meta_json["kv_num"] = 0
                meta_json["value_length"].clear()
                meta_json["tensor_meta"].clear()
                meta_json["key"].clear()
                meta_size = 0

        # write last data
        if batch_size > 0:
            self.mset(keys, tensors)
        # write last meta
        if meta_json["kv_num"] > 0:
            meta_name = os.path.join(
                model_prefix, self.MODEL_META_PREFIX, str(root_json["meta_num"]))
            meta_bytes = json.dumps(meta_json).encode()
            meta_buffer = torch.from_numpy(np.frombuffer(
                meta_bytes, dtype=np.int8).copy())
            self.mset([meta_name], [meta_buffer])
            # update root
            root_json["kv_num_per_meta"].append(meta_json["kv_num"])
            root_json["meta_num"] += 1
        # write root
        logger.info(
            f"write root {model_prefix}, content: {root_json}"
        )
        root_bytes = json.dumps(root_json).encode()
        root_buffer = torch.from_numpy(np.frombuffer(
            root_bytes, dtype=np.int8).copy())
        self.mset([model_prefix], [root_buffer])

        # upload files
        file_list: List[str] = []
        for root, _, files in os.walk(local_model_path):
            for file_name in files:
                # ignore hidden files
                if file_name.startswith("."):
                    continue
                if os.path.splitext(file_name)[1] in (".jpg"):
                    continue
                if os.path.splitext(file_name)[1] not in (".bin", ".pt", ".safetensors"):
                    file_list.append(os.path.join(root, file_name))
        self.upload_model_config(model_name, file_list)
        end_time = time.perf_counter()
        logger.info(
            f"save model on rank{rank} to eic cost {end_time - start_time}s")
        # Preventing follower nodes from upload failures caused by the leader node exiting prematurely
        # in multi node scenarios
        time.sleep(30)

    def _load_tensor(self, state_dict, key: str, tensor: torch.Tensor):

        if key not in state_dict:
            logger.warning(f"Key '{key}' not found in state_dict, replace key.")
            if "self_attn.attn_mha." in key:
                true_key = key.replace("self_attn.attn_mha.", "self_attn.")
                key = true_key

        param_data = state_dict[key].data
        param_shape = state_dict[key].shape
        # If loading with LoRA enabled, additional padding may
        # be added to certain parameters. We only load into a
        # narrowed view of the parameter data.
        for dim, size in enumerate(tensor.shape):
            if size < param_shape[dim]:
                param_data = param_data.narrow(
                    dim, 0, size)
        if tensor.shape != param_shape:
            logger.warning(
                "loading tensor of shape %s into "
                "parameter '%s' of shape %s",
                tensor.shape,
                key,
                param_shape,
            )
        param_data.copy_(tensor)
        state_dict.pop(key)

    def _read_huge_kv(
        self,
        state_dict: Dict[str, torch.Tensor],
        dict_key: str,
        key: str,
        value_length: int,
        tensor_meta: Dict[str, any],
        thread_index,
    ) -> torch.Tensor:
        start_time = time.perf_counter()
        # read tensor
        param_data = state_dict[dict_key].data
        param_shape = state_dict[dict_key].shape
        # If loading with LoRA enabled, additional padding may
        # be added to certain parameters. We only load into a
        # narrowed view of the parameter data.
        for dim, size in enumerate(tensor_meta["shape"]):
            if size < param_shape[dim]:
                param_data = param_data.narrow(
                    dim, 0, size)
        if torch.Size(tensor_meta["shape"]) != param_shape:
            logger.warning(
                "loading tensor of shape %s into "
                "parameter '%s' of shape %s",
                tensor_meta["shape"],
                key,
                param_shape,
            )
        split_num = math.ceil(value_length / self.data_split_size)
        for i in range(split_num):
            split_start_time = time.perf_counter()
            split_key = os.path.join(key, str(i))
            start = i * self.data_split_size
            end = min((i + 1) * self.data_split_size, value_length)
            self.mget([split_key], [self.split_tensors[thread_index][0: end - start]])
            split_end_time = time.perf_counter()
            self.perf_statistics["read_huge_kv_split_s"] = self.perf_statistics.get(
                "read_huge_kv_split_s", 0) + split_end_time - split_start_time
            self.perf_statistics["read_huge_kv_split_ops"] = self.perf_statistics.get(
                "read_huge_kv_split_ops", 0) + 1
            param_data.view(torch.int8).view(value_length)[
            start:end].copy_(self.split_tensors[thread_index][0: end - start])
            load_end_time = time.perf_counter()
            self.perf_statistics["read_huge_kv_copy_split_s"] = self.perf_statistics.get(
                "read_huge_kv_copy_split_s", 0) + load_end_time - split_end_time
        end_time = time.perf_counter()
        self.perf_statistics["read_huge_kv_s"] = self.perf_statistics.get(
            "read_huge_kv_s", 0) + end_time - start_time
        self.perf_statistics["read_huge_kv_ops"] = self.perf_statistics.get(
            "read_huge_kv_ops", 0) + 1
        state_dict.pop(dict_key)
        return

    def _read_kv(
        self,
        keys: List[str],
        tensor_metas: List[Dict[str, any]],
    ) -> List[torch.Tensor]:
        start_time = time.perf_counter()
        tensors: List[torch.Tensor] = []
        for i, meta in enumerate(tensor_metas):
            tensors.append(get_tensor_from_metadata(meta))
        self.mget(keys, tensors)
        end_time = time.perf_counter()
        self.perf_statistics["read_multi_kv_s"] = self.perf_statistics.get(
            "read_multi_kv_s", 0) + end_time - start_time
        self.perf_statistics["read_multi_kv_ops"] = self.perf_statistics.get(
            "read_multi_kv_ops", 0) + 1
        return tensors

    def _read_meta(
        self,
        model_name: str,
    ) -> None:
        start_time = time.perf_counter()
        # read meta and load to dict loader_kv
        for i in range(self.meta_num):
            meta_name = os.path.join(
                model_name, self.meta_prefix, str(i))
            tensors = self.mget([meta_name])
            if len(tensors) == 0:
                raise ValueError(f"Meta {meta_name} not found!")
            meta_bytes = tensors[0].numpy().tobytes()
            meta_json = json.loads(meta_bytes.decode())
            meta_kv_num = meta_json["kv_num"]
            if meta_kv_num != self.kv_per_meta[i]:
                raise ValueError(
                    f"Meta {meta_name} kv num {meta_kv_num} "
                    f"not equal to root kv_per_meta {self.kv_per_meta[i]}")
            for j in range(meta_kv_num):
                key = meta_json["key"][j]
                value_length = meta_json["value_length"][j]
                tensor_meta = meta_json["tensor_meta"][j]
                self.loaded_kv[key] = (value_length, tensor_meta)
        end_time = time.perf_counter()
        self.perf_statistics["read_meta_latency_s"] = end_time - start_time

    def _read_root(
        self,
        root_name: str,
    ) -> None:
        start_time = time.perf_counter()
        tensors = self.mget([root_name])
        if len(tensors) == 0:
            raise ValueError(f"Root {root_name} not found!")
        root_bytes = tensors[0].numpy().tobytes()
        root_json = json.loads(root_bytes.decode())
        logger.info(
            f"read root {root_name}, content: {root_json}"
        )
        self.kv_num = root_json["kv_num"]
        self.meta_num = root_json["meta_num"]
        self.data_split_size = root_json["data_split_size"]
        self.meta_prefix = root_json["meta_prefix_name"]
        self.kv_per_meta = root_json["kv_num_per_meta"]
        end_time = time.perf_counter()
        self.perf_statistics["read_root_latency_s"] = end_time - start_time
        if self.ENABLE_ZERO_COPY:
            if not self.ENABLE_THREADING:
                self.thread_num = 1
            for i in range(self.thread_num):
                zero_copy_ptr = self.allocate_zero_copy_buffer(
                    self.data_split_size)
                ubyte_ptr = ctypes.cast(
                    zero_copy_ptr, ctypes.POINTER(ctypes.c_ubyte))
                byte_array = (
                    ctypes.c_ubyte * self.data_split_size).from_address(ctypes.addressof(ubyte_ptr.contents))
                data_bytes = memoryview(byte_array)
                split_tensor = torch.frombuffer(data_bytes, dtype=torch.int8)
                self.zero_copy_ptrs.append(zero_copy_ptr)
                self.split_tensors.append(split_tensor)
        else:
            if not self.ENABLE_THREADING:
                self.thread_num = 1
            for i in range(self.thread_num):
                split_tensor = torch.empty(
                    self.data_split_size, dtype=torch.int8)
                self.split_tensors.append(split_tensor)

    def _load_eic_worker(self, model_name, state_dict, sub_items: list[tuple[str, Any]], thread_index):
        # iterate all keys
        batch_size = 0
        eic_keys = []
        state_dict_keys = []
        tensor_metas = []
        for key, meta in sub_items:
            # logger.info(f"model_name {model_name} load key {key} meta {meta}")
            value_length = meta[0]
            tensor_meta = meta[1]
            if value_length > self.data_split_size:
                # read and copy tensor to state_dict
                self._read_huge_kv(state_dict, key, os.path.join(
                    model_name, key), value_length, tensor_meta, thread_index)
            else:
                eic_keys.append(os.path.join(model_name, key))
                state_dict_keys.append(key)
                batch_size = batch_size + value_length
                tensor_metas.append(tensor_meta)
                if batch_size > self.MODEL_DATA_SPLIT_SIZE:
                    tensors = self._read_kv(eic_keys, tensor_metas)
                    for i, tensor in enumerate(tensors):
                        self._load_tensor(
                            state_dict, state_dict_keys[i], tensor)
                    # reset
                    batch_size = 0
                    eic_keys.clear()
                    state_dict_keys.clear()
                    tensor_metas.clear()
        if batch_size > 0:
            tensors = self._read_kv(eic_keys, tensor_metas)
            for i, tensor in enumerate(tensors):
                self._load_tensor(state_dict, state_dict_keys[i], tensor)

    def load_model(self, model_prefix, state_dict, rank):
        start_time = time.perf_counter()
        # read root
        self._read_root(model_prefix)
        # read meta
        self._read_meta(model_prefix)
        # iterate all keys
        batch_size = 0
        eic_keys = []
        state_dict_keys = []
        tensor_metas = []
        if self.ENABLE_THREADING:
            items = list(self.loaded_kv.items())
            chunk_size = (len(items) + self.thread_num - 1) // self.thread_num
            threads = []
            for i in range(self.thread_num):
                start = i * chunk_size
                end = min(start + chunk_size, len(items))
                sub_items = items[start:end]
                thread = threading.Thread(target=self._load_eic_worker,
                                          args=(model_prefix, state_dict, sub_items, i))
                threads.append(thread)
                thread.start()
            for thread in threads:
                thread.join()
        else:
            for key, meta in self.loaded_kv.items():
                # logger.info(f"model_prefix {model_prefix} load key {key} meta {meta}")
                value_length = meta[0]
                tensor_meta = meta[1]
                if value_length > self.data_split_size:
                    # read and copy tensor to state_dict
                    self._read_huge_kv(state_dict, key, os.path.join(
                        model_prefix, key), value_length, tensor_meta, 0)
                else:
                    eic_keys.append(os.path.join(model_prefix, key))
                    state_dict_keys.append(key)
                    batch_size = batch_size + value_length
                    tensor_metas.append(tensor_meta)
                    if batch_size > self.MODEL_DATA_SPLIT_SIZE:
                        tensors = self._read_kv(eic_keys, tensor_metas)
                        for i, tensor in enumerate(tensors):
                            self._load_tensor(state_dict, state_dict_keys[i], tensor)
                        # reset
                        batch_size = 0
                        eic_keys.clear()
                        state_dict_keys.clear()
                        tensor_metas.clear()
            if batch_size > 0:
                tensors = self._read_kv(eic_keys, tensor_metas)
                for i, tensor in enumerate(tensors):
                    self._load_tensor(state_dict, state_dict_keys[i], tensor)

        if state_dict:
            raise ValueError(
                f"Missing keys {tuple(state_dict)} in loaded state!")

        if self.ENABLE_ZERO_COPY:
            for i in range(self.thread_num):
                self.release_zero_copy_buffer(
                    self.zero_copy_ptrs[i], self.data_split_size)
            self.split_tensors = []
            self.zero_copy_ptrs = []
        else:
            self.split_tensor = None
        end_time = time.perf_counter()
        self.perf_statistics["load_model_s"] = end_time - start_time
        logger.info(
            f"load model on rank{rank} from eic, perf_stat: {self.perf_statistics}")

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
