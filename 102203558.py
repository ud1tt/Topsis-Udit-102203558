
import pandas as pd
import sys
import numpy as np

def topsis(input_file, weights, impacts, output_file):
    try:
        # Load the data
        data = pd.read_csv(input_file)

        # Validate the input
        if data.shape[1] < 3:
            raise Exception("Input file must have at least three columns.")

        weights = list(map(float, weights.split(',')))
        impacts = impacts.split(',')

        if len(weights) != (data.shape[1] - 1) or len(impacts) != (data.shape[1] - 1):
            raise Exception("Number of weights and impacts must match the number of criteria.")

        if not all(i in ['+', '-'] for i in impacts):
            raise Exception("Impacts must be '+' or '-'.")

        # Separate the criteria and normalize the data
        criteria_data = data.iloc[:, 1:].values
        norm_data = criteria_data / np.sqrt((criteria_data**2).sum(axis=0))

        # Apply weights
        weighted_data = norm_data * weights

        # Determine ideal best and ideal worst
        ideal_best = np.max(weighted_data, axis=0).copy()
        ideal_worst = np.min(weighted_data, axis=0).copy()
        
        for i in range(len(impacts)):
            if impacts[i] == '-':
                ideal_best[i], ideal_worst[i] = ideal_worst[i], ideal_best[i]

        # Calculate the separation measures
        separation_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
        separation_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))

        # Calculate the Topsis score
        topsis_score = separation_worst / (separation_best + separation_worst)

        # Rank the scores
        data['Topsis Score'] = topsis_score
        data['Rank'] = topsis_score.argsort()[::-1].argsort() + 1

        # Save the result to the output file
        data.to_csv(output_file, index=False)
        print(f"Results have been saved to {output_file}")

    except FileNotFoundError:
        print("Input file not found. Please check the file path.")
    except Exception as e:
        print(f"Error: {e}")

# Command-line execution
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
    else:
        _, input_file, weights, impacts, output_file = sys.argv
        topsis(input_file, weights, impacts, output_file)
