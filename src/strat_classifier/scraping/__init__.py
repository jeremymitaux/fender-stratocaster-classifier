"""Data-collection pipeline for the guitar make/model classifier.

Run in order:

1. ``scrape``  — pull electric-guitar listings from Reverb.com, one query per
   model, paginated and de-duplicated by listing id (needs APIFY_TOKEN)
2. ``prepare`` — label listings by make/model and download images into class folders

``download`` is an alternative that grabs raw images per listing (unlabeled).
"""
