import os

runs = [("tested_positive", "positive"), ("tested_negative", "negative"),
        ("can_not_test", "can_not_test"), ("death", "death"), ("cure", "cure_and_prevention")]
for (event, file_map) in runs:
    print("\n\n================== Starting task:", event, "=====================\n\n")
    os.system("python3 sent_model.py --data preproced_" + file_map + "-add_text.json --task " + event )
