import pandas as pd


def extract_markdown_tables(markdown: str) -> list[str]:
    table= []
    for line in markdown.splitlines():
        if is_table_line(line):
            table.append(line)

    return ["\n".join(table)] if table else []

def is_table_line(line: str) -> bool:
    return line is not None and line.strip().startswith("|") and line.strip().endswith("|") and line.count("|") >= 2

def parse_markdown_table(table: str):
    print(table)
    lines = table.split("\n")

    headers = lines[0].strip("|").split("|")
    headers = [header.strip() for header in headers ]
    data = [] 

    # Loop through lines starting from 2
    for line in lines[2:]:
        
        # Break once we hit an empty line
        
        if not line.strip():
            break
            
        cols = line.strip("|").split("|")
        cols = [col.strip() for col in cols]
        #print(f"cols={cols}")
        row = dict(zip(headers, cols))
        data.append(row)
    df = pd.DataFrame(data)
    return df

def rename_headers(df):
        mapping = {'Rérérence': 'reference', 'Quantité': 'quantity', 'Montant': 'pv', 'Prix NHT': 'pvu' }
        df = df.rename(columns=mapping)
        return df

def normalize_supplier_df_to_beluo(df):
    df = rename_headers(df)
    new_df = df.drop(['Libellé de la référence','Prix HT', '%Rem.'], axis=1)
    return new_df.to_json()