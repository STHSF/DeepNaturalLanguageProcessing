import pandas as pd

tupList = [('Eisenstadt', 'Paris','1', '2'), ('London', 'Berlin','1','3'), ('Berlin', 'stuttgat','1', '4'),
           ('Liverpool', 'Southampton','1', '5'),('Tirana', 'Blackpool', '1', '6'),('blackpool', 'tirana','1','7'),
           ('Eisenstadt', 'Paris','1', '2'), ('London', 'Berlin','1','3'), ('Berlin', 'stuttgat','1', '4'),
           ('Paris', 'Lyon','1','8'), ('Manchester', 'Nice','1','10'),('Orleans', 'Madrid','1', '12'),
           ('Lisbon','Stockholm','1','12')]


cities = pd.DataFrame(tupList, columns=['Origin', 'Destination', 'O_Code', 'D_code'])

print(cities)


cities = cities.drop_duplicates().reset_index(drop=True)

print(cities)
