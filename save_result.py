import csv
import os

def save_result(final_result, current_filename):
    csv_filename = f"{current_filename}_final_result_1-5B_finetuning.csv"

    file_exists = os.path.isfile(csv_filename)

    with open(csv_filename, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=final_result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(final_result)