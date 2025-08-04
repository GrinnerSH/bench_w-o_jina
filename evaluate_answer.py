import csv
import numpy as np
from tqdm import tqdm
from eval_grader_post import grade_question

def evaluate_answers():
    input_answers_file = 'my_model_answers_post.csv'
    scores = []

    with open(input_answers_file, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in tqdm(list(reader)): 
            score, extracted_answer, reason = grade_question(
                row['prompt'], 
                row['correct_answer'], 
                row['model_response']
            )
            scores.append(score)

            print(f"\nID: {row.get('id', 'N/A')}, Score: {score}, Reason: {reason}")

    accuracy = np.average(scores) * 100 if scores else 0
    print(f"\n\n===================================")
    print(f"Final Accuracy: {accuracy:.2f}%")
    print(f"===================================")

if __name__ == "__main__":
    evaluate_answers()
