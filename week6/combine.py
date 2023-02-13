import json
import glob

def get_results(dataset:str):
    with open('results/week5/true-results-' + dataset + '.json', 'r') as f:
        result = json.load(f)
    
    return result

def main():
    file_list = glob.glob('results/week5/true-results-*.json')
    results = {}
    for file in file_list:
        print(file.split('/')[-1].split('-')[-1].split('.')[0])

    print(file_list)

if __name__ == '__main__':
    main()

