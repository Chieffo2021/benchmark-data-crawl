import os
import json
import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from typing import List
from pydantic import BaseModel, Field
from crawl4ai.chunking_strategy import SlidingWindowChunking

class IndicatorItem(BaseModel):
    name: str = Field(..., description="Name of the indicator.")
    value: str = Field(..., description="Value of the indicator.")

class BenchmarkItem(BaseModel):
    name: str = Field(..., description="Name of the benchmark.")
    indicators: List[IndicatorItem] = Field(..., description="Indicators of the benchmark.")

class BenchmarkResult(BaseModel):
    benchmarks: List[BenchmarkItem] = Field(..., description="Benchmarks of the benchmark.")

class BenchmarkResultRaw(BaseModel):
    benchmarkIndicatorNames: List[str] = Field(..., description="Indicator names of the benchmark")
    benchmarks: List[List[str]] = Field(..., description="Indicator values of the benchmark.")

async def extract_benchmark_result():
    url = 'https://github.com/inikep/lzbench'

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            word_count_threshold=1,
            excluded_tags=['nav', 'footer'],
            remove_overlay_elements=True,
            exclude_external_links=True,
            exclude_social_media_links=True,
            exclude_external_images=True,
            cache_mode=None,
            extraction_strategy=LLMExtractionStrategy(
                api_base="https://api.302.ai",
                provider="litellm_proxy/llama3.3-70b", # Or use ollama like provider="ollama/nemotron"
                api_token=os.getenv('LLM_API_KEY'),
                schema=BenchmarkResultRaw.model_json_schema(),
                extraction_type="schema",
                # chunk_token_threshold=1000,
                instruction="Extract benchmark results table content"
            ),
            bypass_cache=True,
            # chunking_strategy=SlidingWindowChunking(),
        )

    if result.extracted_content:
        model_fees = json.loads(result.extracted_content)
        print(f"Number of models extracted: {len(model_fees)}")

        with open(".data/result.json", "w", encoding="utf-8") as f:
            json.dump(model_fees, f, indent=2)

asyncio.run(extract_benchmark_result())