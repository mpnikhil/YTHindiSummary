# Youtube Hindi Summary
A Python script to generate summaries of YouTube videos in English and translate them into Hindi using various machine translation models.

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Usage](#usage)
5. [Models Used](#models-used)
6. [Example Output](#example-output)
## Introduction
This script uses the youtube_transcript_api library to extract transcripts from YouTube videos, and then summarizes them using a large language model (LLM). The summary is then translated into Hindi using three different machine translation models: Gemma, Indic, and LLAMA.

## Features
* Extracts transcript from YouTube video
* Summarizes transcript using LLM
* Translates summary into Hindi using three different models:
  * Gemma
  * Indic
  * LLAMA
## Requirements
* Python 3.8+
* nltk library (for tokenization)
* youtube_transcript_api library (for extracting transcripts)
* transformers library (for LLM and machine translation models)
* peft library (for Gemma model)
* IndicTransToolkit library (for Indic model)
## Usage
1. Install required libraries using `pip install -r requirements.txt`
2. Run the script using `python YoutubeHindiSummary.py`
3. Enter the YouTube video link when prompted
4. The script will print out the summary in English and Hindi translations using each of the three models
## Models Used
* **monsterapi/gemma-2-2b-hindi-translator**: PEFT Adaptor for google/gemma-2-2b-it for Eng➡️ Hindi translation.
* **ai4bharat/indictrans2-en-indic-1B**: Seq2seq transformer model for English to Indic language translation.
* **llama3.3**: New state of the art 70B model
