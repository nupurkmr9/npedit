"""Visualization utilities for generating HTML galleries and other visual outputs."""

import html
import os
from typing import Any


def generate_html_gallery(
    output_dir: str,
    asset_metadata_list: list[dict[str, Any]],
    sampling_info: dict[str, Any],
    images_per_row: int | None = None,
    filename: str = "index.html",
) -> None:
    """
    Generate an HTML gallery with images and their associated metadata.

    Args:
        output_dir: Directory where the HTML file will be saved
        asset_metadata_list: List of dictionaries containing image metadata
        sampling_info: Dictionary containing sampling parameters for the header
        images_per_row: Number of images per row in the gallery (defaults to samples_per_prompt * 2)
        filename: Name of the HTML file to generate
    """
    # Determine grid layout
    if images_per_row is None:
        images_per_row = 4

    # Build HTML content
    html_lines: list[str] = []
    html_lines.append("<html><head><meta charset='utf-8'>")

    # CSS styles for responsive gallery
    html_lines.append(
        "<style>\n"
        "body{font-family:sans-serif;margin:0;padding:16px;}\n"
        f".gallery{{display:grid;grid-template-columns:repeat({images_per_row},1fr);gap:16px;}}\n"
        ".gallery figure{margin:0;display:flex;flex-direction:column;}\n"
        ".gallery img{max-width:100%;height:auto;display:block;border-radius:4px;}\n"
        ".gallery figcaption{margin-top:4px;font-size:0.9rem;word-break:break-word;overflow-wrap:anywhere;white-space:normal;}\n"
        "</style>\n"
    )
    html_lines.append("</head><body>")

    # Sampling information header
    html_lines.append("<section style='margin-bottom:24px;'>")
    html_lines.append("  <h2>Sampling Information</h2>")
    html_lines.append("  <ul>")

    for key, value in sampling_info.items():
        if isinstance(value, str):
            escaped_value = html.escape(value, quote=True)
        else:
            escaped_value = str(value)
        html_lines.append(f"    <li><strong>{key}</strong>: {escaped_value}</li>")

    html_lines.append("  </ul>")
    html_lines.append("</section>")

    # Image gallery
    html_lines.append("<div class='gallery'>")

    for metadata in asset_metadata_list:
        path = metadata["path"]
        prompt = metadata["prompt"]
        safe_prompt = html.escape(prompt, quote=True)
        html_lines.append(
            f"<figure><img src='{path}' alt='{safe_prompt}'><figcaption>{safe_prompt}</figcaption></figure>"
        )

    html_lines.append("</div></body></html>")

    # Write the HTML file
    html_path = os.path.join(output_dir, filename)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_lines))
