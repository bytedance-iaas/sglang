"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
from enum import IntEnum, auto
import ipaddress
import logging
import random
import urllib
from http import HTTPStatus
from itertools import chain
from typing import Optional

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.router_args import RouterArgs

try:
    from sglang.srt.tracing.trace import (
        process_tracing_init,
        trace_get_remote_propagate_context,
        trace_req_finish,
        trace_req_start,
        trace_set_thread_info,
        trace_slice_end,
        trace_slice_start,
    )

    trace_package_imported = True
except ImportError:
    trace_package_imported = False

logger = logging.getLogger(__name__)

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class ServerRole(IntEnum):
    ENCODE = auto()
    PREFILL = auto()
    DECODE = auto()
    TEXT = auto()

class MiniLoadBalancer:
    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.encode_urls = [url[0] for url in router_args.encode_urls]
        self.encode_bootstrap_ports = [url[1] for url in router_args.encode_urls]
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.text_urls = router_args.text_urls
        self.otlp_traces_endpoint = router_args.otlp_traces_endpoint
        self.enable_trace = router_args.enable_trace
        if self.enable_trace and not trace_package_imported:
            logger.warning(
                "Tracing is not supported in this environment. Please install sglang."
            )
            self.enable_trace = False

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is only for debugging purposes, it only supports random policy!\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Overriding policy to random")
            router_args.policy = "random"

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        if self.enable_trace:
            process_tracing_init(self.otlp_traces_endpoint, "sglang")
            trace_set_thread_info("Mini lb")
        uvicorn.run(app, host=self.host, port=self.port)

    def select_pair(self):
        if not self.text_urls or not self.encode_urls:
            assert len(self.prefill_urls) > 0, "No prefill servers available"
            assert len(self.decode_urls) > 0, "No decode servers available"
        if self.prefill_urls:
            pidx = random.randint(0, len(self.prefill_urls) - 1)
        else:
            pidx = None
        if self.decode_urls:
            didx = random.randint(0, len(self.decode_urls) - 1)
        else:
            didx = None
        if self.encode_urls:
            eidx = random.randint(0, len(self.encode_urls) - 1)
        else:
            eidx = None
        if self.text_urls:
            tidx = random.randint(0, len(self.text_urls) - 1)
        else:
            tidx = None
        return (
            self.prefill_urls[pidx] if pidx is not None else None,
            self.prefill_bootstrap_ports[pidx] if pidx is not None else None,
            self.decode_urls[didx] if didx is not None else None,
            self.encode_urls[eidx] if eidx is not None else None,
            self.encode_bootstrap_ports[eidx] if eidx is not None else None,
            self.text_urls[tidx] if tidx is not None else None,
        )

    async def get_responses(
        self,
        prefill_server,
        decode_server,
        encode_server,
        text_server,
        endpoint,
        modified_request,
        modified_request_for_prefill,
    ):
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=self.timeout
            )  # Add timeout for request reliability
        ) as session:
            return await self._get_responses_from_session(
                session,
                prefill_server,
                decode_server,
                encode_server,
                text_server,
                endpoint,
                modified_request,
                modified_request_for_prefill,
            )

    async def _get_responses_from_session(
        self,
        session,
        prefill_server,
        decode_server,
        encode_server,
        text_server,
        endpoint,
        modified_request,
        modified_request_for_prefill,
    ):
        tasks_mapping = dict()
        for server_role, server in [
            (ServerRole.PREFILL, prefill_server),
            (ServerRole.DECODE, decode_server),
            (ServerRole.ENCODE, encode_server),
            (ServerRole.TEXT, text_server),
        ]:
            if server:
                if server_role == ServerRole.PREFILL or server_role == ServerRole.TEXT:
                    req = modified_request_for_prefill
                else:
                    req = modified_request
                # print(f"req for {server_role}: {req=}")
                tasks_mapping[server_role] = session.post(
                    f"{server}/{endpoint}", json=req
                )

        # print(f"requests {tasks_mapping.values()=}")

        # Wait for all responses to complete. Prefill should end first.
        responses = await asyncio.gather(*tasks_mapping.values())

        # Extract responses based on server roles
        response_mapping = {}
        response_idx = 0
        for server_role, _ in [
            (ServerRole.PREFILL, prefill_server),
            (ServerRole.DECODE, decode_server),
            (ServerRole.ENCODE, encode_server),
            (ServerRole.TEXT, text_server),
        ]:
            if server_role in tasks_mapping:
                response_mapping[server_role] = responses[response_idx]
                response_idx += 1

        prefill_response = response_mapping.get(ServerRole.PREFILL)
        decode_response = response_mapping.get(ServerRole.DECODE)
        encode_response = response_mapping.get(ServerRole.ENCODE)
        text_response = response_mapping.get(ServerRole.TEXT)
        return prefill_response, decode_response, encode_response, text_response

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint, encode_server=None,
        text_server=None,
        modified_request_for_prefill=None,
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"
        prefill_response, decode_response, encode_response, text_response = (
            await self.get_responses(
                prefill_server,
                decode_server,
                encode_server,
                text_server,
                endpoint,
                modified_request,
                modified_request_for_prefill,
            )
        )

        # async with aiohttp.ClientSession(
        #     timeout=aiohttp.ClientTimeout(
        #         total=self.timeout
        #     )  # Add timeout for request reliability
        # ) as session:
        #     headers = {}
        #     bootstrap_room_list = []
        #     if self.enable_trace:
        #         bootstrap_room_list = (
        #             modified_request["bootstrap_room"]
        #             if isinstance(modified_request["bootstrap_room"], list)
        #             else [modified_request["bootstrap_room"]]
        #         )
        #         trace_context = trace_get_remote_propagate_context(bootstrap_room_list)
        #         headers = {"trace_context": trace_context}

        #     tasks = [
        #         session.post(
        #             f"{prefill_server}/{endpoint}",
        #             json=modified_request,
        #             headers=headers,
        #         ),
        #         session.post(
        #             f"{decode_server}/{endpoint}",
        #             json=modified_request,
        #             headers=headers,
        #         ),
        #     ]

        #     for bootstrap_room in bootstrap_room_list:
        #         trace_slice_end("mini_lb_launch", bootstrap_room, auto_next_anon=True)

        #     # Wait for both responses to complete. Prefill should end first.
        #     prefill_response, decode_response = await asyncio.gather(*tasks)

        if "return_logprob" in modified_request:
            final_response = decode_response
            if prefill_response and decode_response:
                prefill_json, ret_json = await asyncio.gather(
                    prefill_response.json(), decode_response.json()
                )

            # merge `meta_info.input_token_logprobs` from prefill to decode
            if "meta_info" in ret_json and "meta_info" in prefill_json:
                if "input_token_logprobs" in ret_json["meta_info"]:
                    ret_json["meta_info"]["input_token_logprobs"] = (
                        prefill_json["meta_info"]["input_token_logprobs"]
                        + ret_json["meta_info"]["input_token_logprobs"]
                    )
            else:
                # Fallback to decode response only if prefill is not available
                ret_json = await decode_response.json() if decode_response else {}
        else:
            if decode_server:
                final_response = decode_response
            else:
                assert text_server
                print(f"using text response as decode_response")
                final_response = text_response
            ret_json = await decode_response.json() if final_response else {}

        for bootstrap_room in bootstrap_room_list:
            trace_slice_end(
                "wait_PD_finish",
                bootstrap_room,
                thread_finish_flag=True,
            )
            trace_req_finish(bootstrap_room)

        return ORJSONResponse(
            content=ret_json,
            status_code=final_response.status if final_response else 200,
        )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, encode_server=None,
        text_server=None,
        modified_request_for_prefill=None,
        endpoint="generate"
    ):
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        async def stream_results():
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(
                    total=self.timeout
                )  # Add timeout for request reliability
            ) as session:
                # Create the tasks for both prefill and decode requests
                # headers = {}
                # bootstrap_room_list = []
                # if self.enable_trace:
                #     bootstrap_room_list = (
                #         modified_request["bootstrap_room"]
                #         if isinstance(modified_request["bootstrap_room"], list)
                #         else [modified_request["bootstrap_room"]]
                #     )
                #     trace_context = trace_get_remote_propagate_context(
                #         bootstrap_room_list
                #     )
                #     headers = {"trace_context": trace_context}

                # tasks = [
                #     session.post(
                #         f"{prefill_server}/{endpoint}",
                #         json=modified_request,
                #         headers=headers,
                #     ),
                #     session.post(
                #         f"{decode_server}/{endpoint}",
                #         json=modified_request,
                #         headers=headers,
                #     ),
                # ]

                # for bootstrap_room in bootstrap_room_list:
                #     trace_slice_end(
                #         "mini_lb_launch", bootstrap_room, auto_next_anon=True
                #     )
                # # Wait for both responses to complete. Since this is streaming, they return immediately.
                # prefill_response, decode_response = await asyncio.gather(*tasks)

                (
                    prefill_response,
                    decode_response,
                    encode_response,
                    text_response,
                ) = await self._get_responses_from_session(
                    session,
                    prefill_server,
                    decode_server,
                    encode_server,
                    text_server,
                    endpoint,
                    modified_request,
                    modified_request_for_prefill,
                )

                if decode_server:
                    final_response = decode_response
                else:
                    assert text_server
                    # print(f"using text response as decode_response")
                    final_response = text_response

                if modified_request.get("return_logprob", False):
                    # Optimized logprob handling for streaming
                    # 1. Read the entire prefill response first.
                    prefill_body = await prefill_response.read()
                    prefill_response.release()  # Release connection early

                    # 2. Extract the first chunk of prefill to get initial logprobs
                    first_prefill_chunk = (
                        prefill_body.split(b"\n\n")[0].decode("utf-8")[5:].strip()
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)
                    initial_logprobs = first_prefill_chunk_json["meta_info"].get(
                        "input_token_logprobs", []
                    )

                    # 3. Process the decode stream
                    first_chunk = True
                    async for chunk in final_response.content:
                        if first_chunk:
                            # For the first chunk, merge the logprobs
                            decoded_chunk = chunk.decode("utf-8")
                            if (
                                decoded_chunk
                                and decoded_chunk.startswith("data:")
                                and "[DONE]" not in decoded_chunk
                            ):
                                ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                                if "meta_info" in ret_json:
                                    ret_json["meta_info"]["input_token_logprobs"] = (
                                        initial_logprobs
                                        + ret_json["meta_info"].get(
                                            "input_token_logprobs", []
                                        )
                                    )

                                yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                            else:
                                yield chunk
                            first_chunk = False
                        else:
                            # For all subsequent chunks, forward them directly
                            yield chunk
                else:
                    async for chunk in final_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk

            for bootstrap_room in bootstrap_room_list:
                trace_slice_end(
                    "wait_PD_finish",
                    bootstrap_room,
                    thread_finish_flag=True,
                )
                trace_req_finish(bootstrap_room)

        return StreamingResponse(
            stream_results(),
            media_type="text/event-stream",
        )


app = FastAPI()
lb: Optional[MiniLoadBalancer] = None


@app.get("/health")
async def health_check():
    # encode_servers, prefill_servers, decode_servers = (
    #     load_balancer.encode_servers,
    #     load_balancer.prefill_servers,
    #     load_balancer.decode_servers,
    # )
    # async with aiohttp.ClientSession() as session:
    #     # Create the tasks
    #     tasks = []
    #     for server in chain(encode_servers, prefill_servers, decode_servers):
    #         tasks.append(session.post(f"{server}/health_generate"))
    #     for i, response in enumerate(asyncio.as_completed(tasks)):
    #         await response
    # return Response(status_code=200)
    return Response(status_code=200)


@app.get("/health_generate")
async def health_generate():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.get(f"{server}/health_generate"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


# @app.post("/start_profile")
# async def start_profile_async(obj: Optional[ProfileReqInput] = None):
#     encode_servers, prefill_servers, decode_servers, text_servers = (
#         load_balancer.encode_servers,
#         load_balancer.prefill_servers,
#         load_balancer.decode_servers,
#         load_balancer.text_addrs,
#     )
#     async with aiohttp.ClientSession() as session:
#         # Create the tasks
#         tasks = []
#         for server in chain(encode_servers, prefill_servers, decode_servers, text_servers):
#             tasks.append(session.post(f"{server}/start_profile"))
#         for i, response in enumerate(asyncio.as_completed(tasks)):
#             await response
#     return Response(status_code=200)


# @app.post("/stop_profile")
# async def start_profile_async():
#     encode_servers, prefill_servers, decode_servers, text_servers = (
#         load_balancer.encode_servers,
#         load_balancer.prefill_servers,
#         load_balancer.decode_servers,
#         load_balancer.text_addrs,
#     )
#     async with aiohttp.ClientSession() as session:
#         # Create the tasks
#         tasks = []
#         for server in chain(encode_servers, prefill_servers, decode_servers, text_servers):
#             tasks.append(session.post(f"{server}/stop_profile"))
#         for i, response in enumerate(asyncio.as_completed(tasks)):
#             await response
#     return Response(status_code=200)


@app.post("/flush_cache")
async def flush_cache():
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls, lb.encode_urls, lb.text_urls):
            tasks.append(session.post(f"{server}/flush_cache"))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    encode_infos = []
    text_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/get_server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/get_server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])
        for server in lb.encode_urls:
            server_info = await session.get(f"{server}/get_server_info")
            encode_infos.append(await server_info.json())
        for server in lb.text_urls:
            server_info = await session.get(f"{server}/get_server_info")
            text_infos.append(await server_info.json())
    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
            "encode": encode_infos,
            "text": text_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
            "encode": encode_infos,
            "text": text_infos,
        }


@app.get("/get_model_info")
async def get_model_info():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/get_model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError as e:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


def parse_url_as_host(server_addr) -> str:
    """
    Parse and transform prefill_server for bootstrap data
    """
    parsed_url = urllib.parse.urlparse(server_addr)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    return hostname


def modify_bootstrap_info_in_request(
    request_data, bootstrap_server: str, bootstrap_port
):
    """
    In EPD, we have 2 bootstrap servers on encode & prefill
    """
    hostname = parse_url_as_host(bootstrap_server)
    modified_request = request_data.copy()

    batch_size = _get_request_batch_size(modified_request)
    if batch_size is not None:
        modified_request.update(
            {
                "bootstrap_host": [hostname] * batch_size,
                "bootstrap_port": [bootstrap_port] * batch_size,
            }
        )

    else:
        modified_request.update(
            {
                "bootstrap_host": hostname,
                "bootstrap_port": bootstrap_port,
            }
        )
    if "bootstrap_room" not in modified_request:
        modified_request.update(
            {
                "bootstrap_room": (
                    [_generate_bootstrap_room() for _ in range(batch_size)]
                    if batch_size is not None
                    else _generate_bootstrap_room()
                )
            }
        )
    return modified_request


@app.post("/generate")
async def handle_generate_request(request_data: dict):
    prefill_server, bootstrap_port, decode_server, encode_server, encode_bootstrap_port, text_server = lb.select_pair()

    modified_request = modify_bootstrap_info_in_request(
        request_data, prefill_server, bootstrap_port
    )
    if encode_server:
        modified_request_for_prefill = modify_bootstrap_info_in_request(
            modified_request, encode_server, encode_bootstrap_port
        )
    else:
        modified_request_for_prefill = modified_request

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request, prefill_server, decode_server, encode_server=encode_server,
            text_server=text_server,
            modified_request_for_prefill=modified_request_for_prefill,
        )
    else:
        return await lb.generate(
            modified_request, prefill_server, decode_server, encode_server=encode_server,
            text_server=text_server,
            modified_request_for_prefill=modified_request_for_prefill,
        )


async def _forward_to_backend(request_data: dict, endpoint_name: str):
    prefill_server, bootstrap_port, decode_server = lb.select_pair()

    # Parse and transform prefill_server for bootstrap data
    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": hostname,
            "bootstrap_port": bootstrap_port,
            "bootstrap_room": _generate_bootstrap_room(),
        }
    )

    if request_data.get("stream", False):
        return await lb.generate_stream(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )
    else:
        return await lb.generate(
            modified_request,
            prefill_server,
            decode_server,
            endpoint=endpoint_name,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/chat/completions")


@app.post("/v1/completions")
async def handle_completion_request(request_data: dict):
    return await _forward_to_backend(request_data, "v1/completions")


def _generate_bootstrap_room():
    bootstrap_room = random.randint(0, 2**63 - 1)
    if lb.enable_trace:
        trace_req_start(bootstrap_room, bootstrap_room, role="router")
        trace_slice_start("mini_lb_launch", bootstrap_room)
    return bootstrap_room


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
