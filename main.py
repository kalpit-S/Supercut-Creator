from openai import OpenAI
import ffmpeg
import re
import json
import os
from dotenv import load_dotenv
import time
import tempfile
import subprocess
import argparse
import configparser

def read_subtitle_file(file_path):
    """Reads the content of a subtitle file (either SRT or VTT), returns a string."""
    print(f"Reading subtitle file from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def clean_subtitles(subtitle_content, file_type='vtt'):
    """
    Cleans the subtitles based on the file type.

    Args:
        subtitle_content (str): The raw subtitle content.
        file_type (str): The type of the subtitle file ('srt' or 'vtt').

    Returns:
        list: A list of cleaned subtitle segments.
    """
    if file_type == 'vtt':
        return clean_vtt(subtitle_content)
    # TODO: Implement clean_srt function for SRT files
    else:
        raise ValueError("Unsupported file type")

def clean_vtt(vtt_content):
    """ Cleans VTT subtitles and removes duplicates, returns a list of dictionaries containing timestamps and text."""
    print("Converting VTT subtitles to a list of dictionaries")
    lines = vtt_content.split('\n')
    cleaned_data = []
    current_timestamp = None
    seen_texts = set()

    for line in lines:
        if '-->' in line:
            current_timestamp = re.search(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
            if current_timestamp:
                current_timestamp = f"{current_timestamp.group(1)} --> {current_timestamp.group(2)}"
        elif line.strip():
            text = re.sub(r'<\d+:\d+:\d+\.\d+><c>|<c>|</c>', '', line).strip()
            if text and text not in seen_texts:
                cleaned_data.append({"timestamp": current_timestamp, "text": text})
                seen_texts.add(text)
                current_timestamp = None

    return cleaned_data

def convert_to_readable_format(cleaned_subtitles, min_time_diff=1):
    """
    Converts cleaned subtitle data to a more readable format for LLM processing, with each subtitle segment on a new line.

    Args:
        cleaned_subtitles (list): List of subtitle dictionaries with 'timestamp' and 'text'.
        min_time_diff (int): Minimum time difference in seconds to consider timestamps separate.

    Returns:
        str: Formatted subtitle string with each subtitle segment on a new line.
    """
    def convert_to_seconds(timestamp):
        """Converts timestamp to total seconds."""
        hours, minutes, seconds = map(int, timestamp.split(':'))
        return hours * 3600 + minutes * 60 + seconds

    readable_format = ""
    last_timestamp = None
    for subtitle in cleaned_subtitles:
        if subtitle['timestamp']:
            start, end = subtitle['timestamp'].split(' --> ')
            start_seconds = convert_to_seconds(start.split('.')[0])
            if last_timestamp is None or start_seconds - last_timestamp >= min_time_diff:
                readable_format += f"[{start.split('.')[0]}] "
                last_timestamp = start_seconds
        readable_format += subtitle['text'] + "\n"

    return readable_format.strip()

def chunk_text_with_overlap(text, max_chunk_size, overlap):
    """
    Divides text into chunks with a specified maximum size and overlap.

    Args:
        text (str): The text to be chunked.
        max_chunk_size (int): Maximum character length for each chunk.
        overlap (int): Number of characters included at the end of one chunk and the start of the next.

    Returns:
        list: List of text chunks.
    """
    chunks = []
    current_start = 0

    while current_start < len(text):
        end = min(current_start + max_chunk_size, len(text))
        next_timestamp_index = text.find("[", end)
        if next_timestamp_index == -1 or next_timestamp_index > end + overlap:
            next_timestamp_index = end
        chunks.append(text[current_start:next_timestamp_index])
        current_start = next_timestamp_index
        if current_start < end:
            current_start = end
        if current_start + overlap > len(text):
            break

    return chunks

def call_gpt(prompt,system_prompt,selected_model,temp=0.5):
    """Calls the GPT API and returns the response."""
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=selected_model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            },
        ],
        max_tokens=1024,
        temperature=temp,
    )
    return response.choices[0].message.content

def to_rigid_json(llm_response):
    """
    Converts an initial non-rigid LLM response containing time intervals to a specific JSON format.

    Args:
        llm_response (str): The initial non-rigid response from the LLM.

    Returns:
        list: A list of time interval pairs if successful, or None if an error occurs.
    """
    llm_prompt = f"""Extract only the finalized time intervals for the supercut from the following response and format them into a JSON array. The format should strictly be as follows, so the python json.loads function can be used directly on your output:
    [
        ["HH:MM:SS", "HH:MM:SS"],
        ...
    ]
    Original Response:
    {llm_response}"""

    system_prompt = "Format only the finalized time intervals into a JSON array as specified, do not include ''''json'. If unable to process, return 'error'."

    response = call_gpt(llm_prompt, system_prompt, "gpt-3.5-turbo-1106")
    #print(f"RESPONSE FROM RIGID JSON: {response}")
    try:
        response_content = json.loads(response)
        if isinstance(response_content, list) and all(isinstance(pair, list) and len(pair) == 2 for pair in response_content):
            print(f"Finalized time intervals: {response_content}")
            return response_content
        else:
            print("Invalid format in the response.")
            return None
    except Exception as e:
        print(f"Error processing the response: {e}")
        return None

def query_llm(cleaned_subtitles, user_prompt, selected_model="gpt-4"):
    """Queries the LLM and returns the relevant timestamps as a list of lists."""
    
    llm_prompt = f"""
    Before you begin summarizing, please clarify your understanding of the topic related to the prompt '{user_prompt}'.

    The task involves the following steps:

    1. **Topic Identification**: Based on the prompt '{user_prompt}', state your understanding of the main topic or theme to be focused on in the transcript.

    2. **Focused Reading**: Carefully read the transcript, paying special attention to sections that pertain to the identified topic.

    3. **Summarization with Timestamps**: Write a summary of the relevant parts, embedding timestamps to indicate when key points related to '{user_prompt}' are discussed.

    4. **Listing Time Intervals**: After summarizing, list out the specific time intervals corresponding to these key discussions. Each interval should be clearly marked, like 'Start: 00:01:20, End: 00:01:45'.

    5. **Justification of Relevance**: For each time interval listed, provide a brief justification as to why it is relevant to the topic. This is crucial for understanding the context and significance of each segment.

    6. **Contextual Continuity in Segmented Transcripts**: If the beginning or end of a transcript chunk seems relevant but lacks context, assume that the necessary context is in the preceding or following chunk. Include these segments in your summary, ensuring a coherent narrative flow.

    Please proceed if the topic '{user_prompt}' is mentioned in the transcript. If it isn't, you may state that no relevant segments were found.

    Here is the transcript for your analysis:
    ---
    {cleaned_subtitles}
    ---
        Before you begin summarizing, please clarify your understanding of the topic related to the prompt '{user_prompt}'.

    The task involves the following steps:

    1. **Topic Identification**: Based on the prompt '{user_prompt}', state your understanding of the main topic or theme to be focused on in the transcript.

    2. **Focused Reading**: Carefully read the transcript, paying special attention to sections that pertain to the identified topic.

    3. **Summarization with Timestamps**: Write a summary of the relevant parts, embedding timestamps to indicate when key points related to '{user_prompt}' are discussed.

    4. **Listing Time Intervals**: After summarizing, list out the specific time intervals corresponding to these key discussions. Each interval should be clearly marked, like 'Start: 00:01:20, End: 00:01:45'.

    5. **Justification of Relevance**: For each time interval listed, provide a brief justification as to why it is relevant to the topic. This is crucial for understanding the context and significance of each segment.

    6. **Contextual Continuity in Segmented Transcripts**: If the beginning or end of a transcript chunk seems relevant but lacks context, assume that the necessary context is in the preceding or following chunk. Include these segments in your summary, ensuring a coherent narrative flow.

    Please proceed if the topic '{user_prompt}' is mentioned in the transcript. If it isn't, you may state that no relevant segments were found.
    """
    try:
        system_prompt = "You are a helpful assistant."
        response = call_gpt(llm_prompt, system_prompt, selected_model)
    except Exception as e:
        print(f"An error occurred while querying the LLM: {e}")
        return None
    print(f"response from llm: {response}")
    return to_rigid_json(response)

def create_input_file_list(segment_files, list_file):
    with open(list_file, 'w') as f:
        for segment_file in segment_files:
            f.write(f"file '{segment_file}'\n")

def cut_segment(input_video, start_time, end_time, output_file, resolution=None):
    ffmpeg_path = config['FFMPEGPath']
    ffmpeg_command = [ffmpeg_path, '-i', input_video, '-ss', start_time, '-to', end_time]
    if resolution:
        ffmpeg_command.extend(['-vf', f'scale={resolution}'])

    ffmpeg_command.append(output_file)

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while cutting the segment: {e}")

def concatenate_segments(list_file, output_file):
    ffmpeg.input(list_file, format='concat', safe=0).output(output_file, c='copy').run()

def create_supercut_from_segments(segments, video_file, supercut_file,output_resolution=None):
    """Create a supercut video from segments defined as a list of lists."""
    print(f"Segments to process: {segments}")
    if not segments:
        print("No segments provided to process. Exiting the creation of supercut.")
        return  

    temp_folder = tempfile.mkdtemp()

    segment_files = []
    for i, segment in enumerate(segments):
        start_time, end_time = segment
        segment_file = os.path.join(temp_folder, f"segment_{i}.mp4")
        cut_segment(video_file, start_time, end_time, segment_file,resolution=output_resolution)
        segment_files.append(segment_file)
    list_file = os.path.join(temp_folder, 'list.txt')
    create_input_file_list(segment_files, list_file)

    concatenate_segments(list_file, supercut_file)

    for segment_file in segment_files:
        os.remove(segment_file)

#TODO add params to config file
def merge_close_timestamps(timestamps, max_gap=1, min_duration=5.0):
    """
    Merges timestamps that are close to each other and removes short intervals.

    Args:
        timestamps (list): A list of timestamp pairs.
        max_gap (float): Maximum allowed gap in seconds to merge timestamps.
        min_duration (float): Minimum duration of a segment in seconds to be included.

    Returns:
        list: A list of merged and filtered timestamp pairs.
    """
    merged_timestamps = []
    current_start, current_end = timestamps[0]

    def convert_to_seconds(t):
        h, m, s = map(float, t.split(':'))
        return h * 3600 + m * 60 + s

    for start, end in timestamps[1:]:
        if convert_to_seconds(start) - convert_to_seconds(current_end) <= max_gap:
            current_end = end  # Extend the current timestamp
        else:
            if convert_to_seconds(current_end) - convert_to_seconds(current_start) >= min_duration:
                merged_timestamps.append([current_start, current_end])
            current_start, current_end = start, end

    # Check the last segment
    if convert_to_seconds(current_end) - convert_to_seconds(current_start) >= min_duration:
        merged_timestamps.append([current_start, current_end])

    return merged_timestamps


# Function to read and return config values
def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config['DEFAULT']

config = get_config()

def validate_files(vtt_file, video_file):
    """Check if files exist."""
    if not os.path.exists(vtt_file):
        raise FileNotFoundError(f"Subtitle file not found: {vtt_file}")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"Video file not found: {video_file}")

def main():
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(description="Create a supercut video based on specific topics in a video.")
    parser.add_argument("video_name", help="Name of the video file (without extension)")
    parser.add_argument("user_prompt", help="Topic or theme to focus on in the video")
    parser.add_argument("--output_file", help="Name for the output video file (without extension)", default="output")
    parser.add_argument("--config", help="Path to the configuration file", default="config.ini")
    parser.add_argument("--model", help="Selected GPT model", choices=["gpt-4-1106","Any other completions model offered by OpenAI"], default="gpt-3.5-turbo-1106")
    parser.add_argument("--resolution", help="Output resolution in the format WIDTH:HEIGHT", default=None)
    
    args = parser.parse_args()

    # If no output file name is provided, use the video name with "_supercut" appended
    if args.output_file is None:
        args.output_file = args.video_name + "_supercut"

    vtt_file = os.path.join(config['SubtitlesFolderPath'], args.video_name + '.vtt')
    video_file = os.path.join(config['VideoFolderPath'], args.video_name + '.mp4')
    output_file = os.path.join(config['OutputFolderPath'], args.output_file + '.mp4')

    # Validate file existence
    validate_files(vtt_file, video_file)

    # Process subtitles and video
    subtitles = read_subtitle_file(vtt_file)
    cleaned_subtitles = clean_subtitles(subtitles, 'vtt')
    readable_subtitles = convert_to_readable_format(cleaned_subtitles)
    subtitle_chunks = chunk_text_with_overlap(readable_subtitles, max_chunk_size=4096, overlap=500)

    master_timestamps = []
    for i, chunk in enumerate(subtitle_chunks):
        time.sleep(5)
        print(f"Processing chunk {i+1} of {len(subtitle_chunks)}")
        try:
            chunk_timestamps = query_llm(chunk, args.user_prompt, args.model)
            if chunk_timestamps:
                master_timestamps.extend(chunk_timestamps)
        except Exception as e:
            print(f"Error querying LLM: {e}")
            break

    unique_timestamps = merge_close_timestamps(master_timestamps, max_gap=5, min_duration=5)
    create_supercut_from_segments(unique_timestamps, video_file, output_file,args.resolution)

if __name__ == "__main__":
    main()


