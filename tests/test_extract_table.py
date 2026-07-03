

import sys

from document_intelligence.parsing.markdown_tables import extract_markdown_tables,parse_markdown_table, normalize_supplier_df_to_beluo

markdown = """

Le délai de livraison est donné à titre indicatif et n'est pas sujet à des pénalités

| Rérérence   | Libellé de la référence                                         | Quantité   | Prix HT   | %Rem.   | Prix NHT   | Montant    |
|-------------|-----------------------------------------------------------------|------------|-----------|---------|------------|------------|
| AR0V240     | U-1000 AR2V 1x240 GL DISPONIBLES EN 2 LONGUEURS : 1560ML+1305ML | ML 2 865   | 4,011€    | 0,00    | 4,011€     | 11 491,51€ |
| AR0V70      | U-1000 AR2V 1x70 GL DISPONIBLE                                  | ML 955     | 1,302€    | 0,00    | 1,302€     | 1 243,41€  |
titre indicatif et n'es
"""



tables = extract_markdown_tables(markdown)

#print(len(tables))
#print(tables[0])
df = parse_markdown_table(tables[0])

new_df = normalize_supplier_df_to_beluo(df)
print("*"*100)
print(new_df)
