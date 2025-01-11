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
    benchmarkDescription: List[str] = Field(..., description="Description of the benchmarks, which may include software names and versions. This field is used for the table row titles.")
    # benchmarkIndicatorNames: List[str] = Field(..., description="Names of the benchmark indicators, specifying the indicators to test performance. This field is used for the table's first column.")
    benchmarks: List[List[str]] = Field(..., description="Values of the benchmark indicators, representing the performance results. These values are used for the table cell contents. The first content of values is Names of the benchmark indicators, specifying the indicators to test performance. This field is used for the table's first column.")

class BenchmarkSchemaDescription(BaseModel):
    description: str = Field(..., description="Description of the benchmark schema.")

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
                schema=BenchmarkSchemaDescription.model_json_schema(),
                extraction_type="schema",
                # chunk_token_threshold=1000,
                instruction="""
                Find the benchmark data from the given content, which may include software name, software version, source code project, test time, performance indicators, indicator test values, etc. Find these related contents, summarize the way the related contents are provided into a pattern that includes all relevant information, and give 2-3 examples, the sample content should be as consistent as possible with the original content format.
                The following <example> tag gives you an example of what you would return:
                <example>
                {
                 description: "
                    Content text is provided in the form of description + table
                    Description: Contains xx software, uses 1 core of Intel Core i7-8700K, runs in Ubuntu 18.04.3 64-bit environment, the test time is 2020-12-1, and the source code project is archived on [repo](https://github.com/inikep/lzbench).
                    Table contents:
                      The header provides the performance indicators of the test, example: | Compressor name | Compress. | Decompress. | Compr.size | Ratio |
                      The first cell of the table row provides the name of the performance indicator of the test, and the values of the remaining cells provide the values of the performance indicators of the test, example: | memcpy | 10362 MB/s | 10790 MB/s | 211947520 | 100.00 |
                      Example 1：
                      | Compressor name | Compress. | Decompress. | Compr.size | Ratio |
                      | memcpy | 10362 MB/s | 10790 MB/s | 211947520 | 100.00 |
                      Example 2：
                      | Compressor name | Compress. | Decompress. | Compr.size | Ratio |
                      | blosclz 2.0.0 -1 | 2342 MB/s | 12355 MB/s | 1234444435 | 100.00 |
                "
                }
                </example>
                restriction:
                Instead of extracting all the benchmark results which would make the content too long, we just need to extract the patterns. !Important!
                """
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