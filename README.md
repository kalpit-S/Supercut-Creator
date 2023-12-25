# Supercut Creator

Supercut Creator is an innovative Python tool designed to automatically generate a supercut video from existing video content, guided by the user's interest. Utilizing advanced Large Language Models (LLMs) for subtitle analysis, it identifies and compiles relevant video segments into a cohesive supercut.

## Features

- Automated extraction of time intervals relevant to a specified topic from video subtitles.
- Generation of a supercut video based on identified segments.
- Customizable output resolution and model selection for enhanced flexibility.

## Prerequisites

Before starting, ensure you have:

- Python 3.x installed.
- FFmpeg installed and set in your system's PATH.
- An OpenAI API key for GPT model usage.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kalpits/Supercut-Creator.git
   cd Supercut-Creator
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Rename `config.ini.example` to `config.ini`.
2. Update the configuration file with your paths for videos, subtitles, output directories, and FFmpeg installation.
3. Set your OpenAI API key in an `.env` file or as an environment variable.

## Usage

1. **Prepare Video and Subtitles**:
   - Place your `.mp4` video and `.vtt` subtitle files in their respective directories as defined in `config.ini`.
   - Ensure the video and subtitle files are named identically.

2. **Execute the Script**:
   ```bash
   python supercut_creator.py [video_name] [topic] [--output_file OUTPUT_FILE] [--config CONFIG_PATH] [--model MODEL_NAME] [--resolution RESOLUTION]
   ```
   Parameters:
   - `video_name`: Base name of the video file (without extension).
   - `topic`: Desired topic for the supercut.
   - `--output_file`: Optional. Name for the output video file (without extension). Default: `[videoname] super cut`.
   - `--config`: Optional. Path to the configuration file. Default: `config.ini`.
   - `--model`: Optional. GPT model choice. Default: `gpt-3.5-turbo-1106`.
   - `--resolution`: Optional. Desired output resolution in `WIDTH:HEIGHT` format.

3. **Access the Supercut**:
   - The final supercut video will be stored in the designated output directory.

## Example

Creating a supercut from a video titled `"Talk about LLMs"`, focusing on `"Speeding up transformer models"`, and naming the output `transformer_speedup_supercut`:

```bash
python supercut_creator.py "Talk about LLMs" "Speeding up transformer models" --output_file transformer_speedup_supercut
```

Ensure the corresponding `.vtt` subtitle file is in your subtitles folder.

## License

Distributed under the MIT License. 

## How It Works

The Supercut Creator script simplifies the process of creating a themed supercut video. It works in several key stages:

### 1. Subtitle Processing

- **Reading Subtitles**: The script starts by reading the VTT subtitle file of the target video.
- **Cleaning Subtitles**: Cleaning and formatting subtitles to extract only the necessary text, maximizing efficiency and reducing token count for LLM processing.

### 2. Language Model Integration

- **Chunking Subtitles**: Subtitles are chunked, considering token limits of LLMs and adding overlap for context.
- **Querying LLM Model**: Each chunk is sent to an LLM with the user-specified topic to find relevant segments.
- **Timestamp Extraction**: A follow-up LLM call converts the model's response into a structured JSON format, extracting precise time intervals.

### 3. Video Processing

- **Cutting Segments**: FFmpeg cuts the video segments based on the identified timestamps.
- **Merging Segments**: These segments are seamlessly concatenated to form the final supercut.

### 4. Flexibility and Customization

- The script offers extensive configurability, including input/output paths, video resolution, and LLM choice, through command-line arguments and a `config.ini` file.

### 5. Practical Use Cases

- Ideal for content creators, educators, and researchers for thematic content compilation from lengthy videos.
- Demonstrates a practical application of combining NLP and video processing for automated content creation.


# Possible Future Updates

- **Cleaner User Feedback**: Cleaning up print statements and giving time estimates.
- **Support for SRT Subtitle Format**
- **Improved Prompting**: Still room to improve performance through prompting / parameter tuning
- **More Config File Options**
- **Adding Support for Google Gemini**: They are currently giving 60 free requests per minute for Gemini Pro!
- **Adding Support for Together AI** API: Allows for the use of hundreds of open source models
- **Relevancy Checks for Segments**: Right before cutting the video segments, making sure if the segments actually fit with the topic (thinking of using a smaller model here).
- **Making a GUI**: Making it easier to use
- **Generating Own Subtitles with "Insanely-fast-whisper"**: Moving towards generating subtitles directly, no subtitle file needed.
- **Support for Any Subtitle Format**: Basically having the LLM on the fly generate code to extract subtitles in the format needed to work with the functions.
