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
    return new_df


def split_quantity_unit(quantity_raw: str) -> tuple[str, int]:
    value = quantity_raw.strip()
    parts = value.split()

    if len(parts) < 2:
        raise ValueError(f"Format quantité/unité invalide : {quantity_raw}")

    unit = parts[0]
    quantity_as_text = "".join(parts[1:])
    quantity = int(quantity_as_text)

    return unit, quantity

def parse_price(price_raw: str) -> float:
    value = price_raw.strip()
    value = value.replace("€", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")

    return float(value)

def normalize_quantity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    quantity_unit = df["quantity"].apply(split_quantity_unit)

    df["unit"] = quantity_unit.apply(lambda x: x[0])
    df["quantity"] = quantity_unit.apply(lambda x: x[1])

    return df


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pvu"] = df["pvu"].apply(parse_price)
    df["pv"] = df["pv"].apply(parse_price)

    return df


def to_beluo_json(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")