import os
import csv
import base64
import sys
import json
import datetime
import argparse
from tqdm import tqdm
from my_model import RRSMAgent

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        # MODIFIED: Ensure the log file is flushed immediately to see real-time progress
        self.log = open(filename, 'a', encoding='utf-8', buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        # self.flush() is not needed with line buffering (buffering=1)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def xor_decrypt(data: bytes, key: str) -> bytes:
    """
    XOR decrypt data with a key
    """
    key_bytes = key.encode('utf-8')
    return bytes([data[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data))])

def generate_answers(dataset_name: str):
    input_dataset = f'data/{dataset_name}.csv'
    output_csv_filename = f'my_model_answers_pre_{dataset_name}.csv'
    raw_log_filename = f'my_model_raw_responses_pre_{dataset_name}.log'

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"pre_log_generate_answers_{dataset_name}_{timestamp}.txt"
    sys.stdout = Logger(log_filename, sys.stdout)
    sys.stderr = sys.stdout # Redirect stderr to the same logger

    questions = []
    print(f"Reading from: {input_dataset}")
    try:
        with open(input_dataset, mode='r', encoding='utf-8-sig') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("canary"):
                    try:
                        key = row["canary"]
                        row["prompt"] = xor_decrypt(base64.b64decode(row["prompt"]), key).decode('utf-8')
                        row["answer"] = xor_decrypt(base64.b64decode(row["answer"]), key).decode('utf-8')
                        questions.append(row)
                    except Exception as e:
                        print(f"Skipping row due to decryption error: {e}")
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_dataset}")
        return

    # agent = RRSMAgent(max_turns=15)
    agent = RRSMAgent(max_turns=25)

    with open(output_csv_filename, mode='w', newline='', encoding='utf-8') as csv_file, \
         open(raw_log_filename, mode='w', encoding='utf-8') as log_file:

        fieldnames = ["id", "prompt", "correct_answer", "model_response", "sources"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        for question in tqdm(questions, desc=f"Generating Answers for {dataset_name}"):
            question_id = question.get('id', '')
            prompt = question.get('prompt', '')
            correct_answer = question.get('answer', '')
            
            model_response = "Error: Agent did not produce a valid response."
            sources = []

            print(f"\n\n{'='*30}\nProcessing Question ID: {question_id}\nPrompt: {prompt}\n{'='*30}")
            log_file.write(f"\n{'='*20} Question ID: {question_id} {'='*20}\n")
            log_file.write(f"Prompt: {prompt}\n")

            try:
                result = agent.run(prompt)
                
                if isinstance(result, dict):
                    model_response = result.get("answer", "Agent returned a dictionary without an 'answer' key.")
                    sources = result.get("sources", [])
                elif isinstance(result, str):
                    model_response = f"Agent finished with a message: {result}"
                    sources = []

            except Exception as e:
                # This catches unexpected errors within the agent.run() call itself.
                print(f"An unexpected error occurred for question ID {question_id}: {e}")
                log_file.write(f"FATAL ERROR during agent.run: {e}\n")
                model_response = f"FATAL ERROR: {e}"
                sources = []

            csv_writer.writerow({
                "id": question_id,
                "prompt": prompt,
                "correct_answer": correct_answer,
                "model_response": model_response,
                "sources": json.dumps(sources, ensure_ascii=False) # Storing sources as a JSON string
            })

            log_file.write(f"Final Model Response: {model_response}\n")
            log_file.write(f"Sources: {json.dumps(sources, ensure_ascii=False)}\n")
            log_file.flush()

    print(f"\nAnswers generated and saved to {output_csv_filename}")
    print(f"Raw responses logged to {raw_log_filename}")
    print(f"Detailed execution log saved to {log_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate answers for a given dataset using the RRSMAgent."
    )
    parser.add_argument(
        "dataset_name",
        type=str,
        help="The name of the dataset to process (e.g., DeepSearch_part1)"
    )
    args = parser.parse_args()

    generate_answers(args.dataset_name)
