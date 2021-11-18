import pandas as pd
import sqlite3
# from NCM import NCM

# base = NCM(None)
# base.data = pd.read_csv("produtos.csv")
# base.generalizeString(column='descp')
# base.data = base.data['descp']
# base.data = base.data.dropna()
# base.save("searchingProducts.csv")

from Request import RequestSEFAZ
request = RequestSEFAZ()

# banco de dados
database = sqlite3.connect("products.db")
cursor = database.cursor()
# criando a tabela de nomes e atributos
cursor.execute("CREATE TABLE IF NOT EXISTS products ("
    +"gtin TEXT NOT NULL,"
    +"descp TEXT NOT NULL PRIMARY KEY,"
    +"ncm TEXT NOT NULL,"
    +"preco REAL NOT NULL"
    +")")

products = pd.read_csv("searchingProducts.csv")
for product in products['descp']:
    data = request.request(product)
    for item in data.json():
        cursor.execute("INSERT OR REPLACE INTO products (gtin, descp, ncm, preco)"
            +" VALUES (?,?,?,?)",
            (item['codGetin'],item['dscProduto'],item['codNcm'],item['valUnitarioUltimaVenda']))
        
        database.commit()