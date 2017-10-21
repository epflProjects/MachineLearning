import csv
from collections import defaultdict
import os

file_to_cluster = 'train'

for filename in os.listdir():
  if filename.endswith(file_to_cluster + '.csv'): 
    rows = list()
    attributes = []

    with open(filename, 'r') as f:
      reader = csv.reader(f)
      attributes = next(reader)

    with open(filename, 'r') as f:
      dictReader = csv.DictReader(f)
      for row in dictReader:
        rows.append(row)

    for indice in range(4):
      with open(file_to_cluster + '_PRI_jet_num_' + str(indice) + '.csv', 'w') as new_file:
        dictWriter = csv.DictWriter(new_file, attributes)
        dictWriter.writeheader()
        for row in rows:
          newRow = {}
          if row['PRI_jet_num'] == str(indice):
            for attr in attributes:
              newRow.update({attr: row[attr]})
            dictWriter.writerow(newRow)