import csv

class CSVWriter:
    def __init__(self, file_name, reward_history):
        self.file_name = file_name
        self.reward_history = reward_history
    
    def write(self):
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.writer(file)

            for i, row in enumerate(self.reward_history, start=1):
                if i > 40:
                    writer.writerow([i] + [row] + [self.avg_prev(i, 50)])
                else:
                    writer.writerow([i] + [row] + [0])

            print(f"Reward history written to {self.file_name}")

    # returns average of previous 10 entries
    def avg_prev(self, i, j):
        preceeding = self.reward_history[i-j:i]
        average = sum(preceeding)/j
        return average


