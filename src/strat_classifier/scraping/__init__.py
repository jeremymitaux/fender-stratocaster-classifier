"""Data-collection pipeline for the Stratocaster classifier.

Run in order:

1. ``scrape``  — pull Stratocaster listings from Reverb.com (needs APIFY_TOKEN)
2. ``prepare`` — label listings by origin and download images into class folders

``download`` is an alternative that grabs raw images per listing (unlabeled).
"""
