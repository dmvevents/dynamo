#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

from dynamo.frontend.sglang_processor import SglangProcessor
from dynamo.frontend.vllm_processor import VllmProcessor


async def _empty_stream():
    if False:
        yield {}


class DummyRouter:
    def __init__(self):
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return _empty_stream()


class DummyOutputProcessor:
    def __init__(self):
        self.request_states = {}

    def add_request(self, *_args):
        return None


def test_vllm_processor_forwards_backend_instance_id_to_kv_router():
    async def run():
        router = DummyRouter()
        processor = VllmProcessor(
            tokenizer=SimpleNamespace(eos_token_ids=[1]),
            input_processor=None,
            router=router,
            output_processor=DummyOutputProcessor(),
            tool_parser_class=None,
            reasoning_parser_class=None,
        )
        processor.is_kv_router = True

        async for _ in processor._generate_and_stream(
            request_id="req-1",
            request={"model": "test-model", "nvext": {"backend_instance_id": 1234}},
            dynamo_preproc={
                "model": "test-model",
                "stop_conditions": {},
                "sampling_options": {},
                "output_options": {},
            },
            tokens=[1, 2, 3],
            vllm_preproc=SimpleNamespace(request_id="engine-1"),
            post=SimpleNamespace(),
        ):
            pass

        assert len(router.calls) == 1
        assert router.calls[0]["worker_id"] == 1234

    asyncio.run(run())


def test_sglang_processor_prefers_decode_worker_id_for_kv_router():
    async def run():
        router = DummyRouter()
        processor = SglangProcessor(
            tokenizer=None,
            router=router,
            tool_call_parser_name=None,
            reasoning_parser_name=None,
            eos_token_id=None,
        )
        processor.is_kv_router = True

        async for _ in processor._generate_and_stream(
            request_id="req-1",
            request={
                "model": "test-model",
                "nvext": {
                    "backend_instance_id": 1234,
                    "decode_worker_id": 5678,
                },
            },
            dynamo_preproc={
                "model": "test-model",
                "stop_conditions": {},
                "sampling_options": {},
                "output_options": {},
            },
            tokens=[1, 2, 3],
            post=SimpleNamespace(),
        ):
            pass

        assert len(router.calls) == 1
        assert router.calls[0]["worker_id"] == 5678

    asyncio.run(run())
