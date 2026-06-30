
# Schema du pipeline

PDF
↓
OCR / extraction texte
↓
pré-processing
↓
tokenization
↓
CamemBERT
↓
NER
↓
entités
↓
validation métier
↓
JSON

## OCR
Technologie capale de convertir des images ou des documents imprimés en texte numérique modifiable

## Pre-processing
Préparation  des données : nettoyage des textes , normalisation des formats avant de passer au modèle

## CamemBERT
Modèle pré-entrainée en langue française pour comprendre le sens général du document (devis, facture,...)

## NER
Ajouter un label aux mots

## Entités 
Sortie du NER

## Validation métier
Permet de vérifier la sortie de NER et de detecter les erreurs eventuels

## JSON
La sortie