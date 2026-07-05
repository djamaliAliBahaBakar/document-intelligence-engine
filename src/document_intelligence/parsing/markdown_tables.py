import pandas as pd

from .header_mapping import map_header
from document_intelligence.parsing.header_mapping import HEADER_MAPPING


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





def normalize_header(header: str) -> str:
    return header.strip().lower()


def rename_headers(df):

    new_columns = {}

    for column in df.columns:

        normalized = normalize_header(column)

        if normalized in HEADER_MAPPING:
            new_columns[column] = HEADER_MAPPING[normalized]
        else:
            new_columns[column] = column

    return df.rename(columns=new_columns)

def normalize_supplier_df_to_beluo(df):
    df = rename_headers(df)
    print(df.columns.tolist())
    columns = [
        "reference",
        "description",
        "unit",
        "quantity",
        "pvu",
        "pv",
    ]

    result = {}

    for column in columns:
        if column in df.columns:
            result[column] = df[column]

    return pd.DataFrame(result)


def split_quantity_unit(quantity_raw: str) -> tuple[str, int]:
    value = quantity_raw.strip()
    parts = value.split()

    if len(parts) < 2:
        raise ValueError(f"Format quantité/unité invalide : {quantity_raw}")

    unit = parts[0]
    quantity_as_text = "".join(parts[1:])
    quantity = int(quantity_as_text)

    return unit, quantity


def parse_french_number(value) -> float:
    value = str(value).strip()
    value = value.replace("€", "")
    value = value.replace(" ", "")
    value = value.replace(",", ".")

    if value == "":
        raise ValueError("Nombre vide")

    return float(value)

def normalize_quantity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "quantity" not in df.columns:
        raise ValueError("Colonne obligatoire manquante : quantity")

    # Si la colonne unit n'existe pas, on applique la valeur par défaut Beluo
    if "unit" not in df.columns:
        df["unit"] = "Unité"

    df["unit"] = (
        df["unit"]
        .fillna("Unité")
        .replace("", "Unité")
        .astype(str)
        .str.strip()
    )

    # Cas A : quantité déjà numérique : "1", "1,00", "89,00"
    quantity_as_text = (
        df["quantity"]
        .astype(str)
        .str.replace(" ", "", regex=False)
    )

    contains_letters = quantity_as_text.str.contains(r"[A-Za-z]", regex=True).any()

    if not contains_letters:
        df["quantity"] = df["quantity"].apply(parse_french_number)
        return df

    # Cas B : quantité + unité fusionnées : "ML 2 865"
    quantity_unit = df["quantity"].apply(split_quantity_unit)

    df["unit"] = quantity_unit.apply(lambda x: x[0])
    df["quantity"] = quantity_unit.apply(lambda x: x[1])

    return df


def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pvu"] = df["pvu"].apply(parse_french_number)
    df["pv"] = df["pv"].apply(parse_french_number)

    return df


def to_beluo_json(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


def normalize_header(header: str) -> str:
    """Normalise un header pour faciliter les comparaisons."""
    return header.strip().lower()






def find_quote_table(markdown: str) -> str:
    """
    Retourne uniquement le tableau contenant les lignes de devis.
    """

    lines = markdown.splitlines()

    start = None

    # Recherche du header du tableau métier
    for i, line in enumerate(lines):

        if not is_table_line(line):
            continue

        headers = [
            h.strip()
            for h in line.strip("|").split("|")
        ]

        mapped = [
            map_header(h)
            for h in headers
        ]

        mapped = [m for m in mapped if m is not None]

        required = {"quantity", "pvu", "pv"}

        if required.issubset(set(mapped)):
            start = i
            break

    if start is None:
        raise ValueError("Aucun tableau métier trouvé.")

    table = []

    for line in lines[start:]:

        if not is_table_line(line):
            break

        headers = [
            h.strip()
            for h in line.strip("|").split("|")
        ]

        mapped = [
            map_header(h)
            for h in headers
        ]

        mapped = [m for m in mapped if m is not None]

        # Nouveau header rencontré
        if (
            len(table) > 2
            and required.issubset(set(mapped))
        ):
            break

        table.append(line)

    return "\n".join(table)


def keep_business_rows(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["quantity", "pvu", "pv"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Colonne obligatoire manquante : {column}")

    return df[
        (df["quantity"] > 0)
        & (df["pvu"] > 0)
        & (df["pv"] > 0)
    ].copy()