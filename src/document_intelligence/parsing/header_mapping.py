HEADER_MAPPING = {
    # Référence
    "rérérence": "reference",
    "référence": "reference",
    "reference": "reference",
    "code article": "reference",

    # Description
    "désignation": "description",
    "designation": "description",
    "libellé": "description",
    "libellé de la référence": "description",
    "description": "description",

    # Unité
    "u": "unit",
    "unité": "unit",
    "unite": "unit",

    # Quantité
    "quantité": "quantity",
    "qté": "quantity",
    "qte": "quantity",

   
    # Prix de vente unitaire Beluo
    "prix nht": "pvu",
    "p.u. ht": "pvu",
    "prix unit.": "pvu",

    # Prix de vente Beluo
    "montant": "pv",
    "t. ht": "pv",
    "total ht": "pv",
    "prix total": "pv",

    "désignaƟon": "description",
    "désignation": "description",

    "description": "description",
    "u.": "unit",
    "p.u. ht": "pvu",
    "montant ht": "pv",
}

from .header_mapping import HEADER_MAPPING


def normalize_header(header: str) -> str:
    return header.strip().lower()


def map_header(header: str) -> str | None:
    header = normalize_header(header)
    return HEADER_MAPPING.get(header)

def is_quote_header(headers: list[str]) -> bool:

    mapped = [
        map_header(h)
        for h in headers
    ]

    mapped = [h for h in mapped if h is not None]

    required = {
        "quantity",
        "pvu",
        "pv",
    }

    return required.issubset(set(mapped))