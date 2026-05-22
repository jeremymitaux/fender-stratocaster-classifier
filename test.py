import os

from apify_client import ApifyClient # type: ignore

client = ApifyClient(os.environ["APIFY_TOKEN"])  # export APIFY_TOKEN=... before running

# Input for the scraper
run_input = {
    "startUrls": [
        {"url": "https://reverb.com/marketplace?query=fender%20stratocaster"}
    ],
    "maxItems": 10
}

# Run the actor
run = client.actor("parseforge/reverb-com-scraper").call(
    run_input=run_input
)

# Get results
for item in client.dataset(run.default_dataset_id).iterate_items():
    print(item)
    

    