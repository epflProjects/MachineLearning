import csv
with open("prediction.csv","r") as source:
    rdr= csv.reader(source)
    with open("result.csv","w") as result:
        wtr= csv.writer(result)
        wtr.writerow(("Id", "Prediction"))
        for r in rdr:
            r0 = r[0].partition(" ")[0]
            r1 = r[1]
            if float(r1) == 0.0:
                r1 = str(1)
            else:
                r1 = str(-1)
            wtr.writerow((r0, r1))
