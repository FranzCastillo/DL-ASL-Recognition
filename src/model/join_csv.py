import csv
import os

DATA_DIR = 'data'
OUTPUT = 'data/ALL_DATA.csv'


def main():
    data = []
    got_header = False
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            path = f'{DATA_DIR}/{file}'
            with open(path, mode='r') as f:
                if path == OUTPUT:
                    continue

                reader = csv.reader(f)
                # Skip the header for all files except the first one
                data.extend(list(reader)[1:]) if got_header else data.extend(list(reader))
                got_header = True

    # Write the data to a new CSV file
    with open(OUTPUT, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)


if __name__ == '__main__':
    main()
