import pandas as pd
import sys
import numpy as np

def perform_topsis(input_file, weights, impacts, output_file):
    try:
        data = pd.read_csv(input_file)

        if data.shape[1] < 3:
            raise Exception("Input file must have at least three columns.")

        weights = list(map(float, weights.split(',')))
        impacts = impacts.split(',')

        if len(weights) != (data.shape[1] - 1) or len(impacts) != (data.shape[1] - 1):
            raise Exception("Number of weights and impacts must match the number of criteria.")

        if not all(i in ['+', '-'] for i in impacts):
            raise Exception("Impacts must be '+' or '-'.")

        criteria_values = data.iloc[:, 1:].values
        normalized_values = criteria_values / np.sqrt((criteria_values**2).sum(axis=0))

        weighted_values = normalized_values * weights

        ideal_best = np.max(weighted_values, axis=0).copy()
        ideal_worst = np.min(weighted_values, axis=0).copy()

        for i in range(len(impacts)):
            if impacts[i] == '-':
                ideal_best[i], ideal_worst[i] = ideal_worst[i], ideal_best[i]

        separation_best = np.sqrt(((weighted_values - ideal_best)**2).sum(axis=1))
        separation_worst = np.sqrt(((weighted_values - ideal_worst)**2).sum(axis=1))

        scores = separation_worst / (separation_best + separation_worst)

        data['Topsis Score'] = scores
        data['Rank'] = scores.argsort()[::-1].argsort() + 1

        data.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

    except FileNotFoundError:
        print("Input file not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        _, input_file, weights, impacts, output_file = sys.argv
        perform_topsis(input_file, weights, impacts, output_file)
