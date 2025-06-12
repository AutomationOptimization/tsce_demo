"""Retrieve compound data from ChEMBL."""

from __future__ import annotations

import requests


class ChEMBLTool:
    """Return SMILES and activities for a ChEMBL ID."""

    endpoint = "https://www.ebi.ac.uk/chembl/api/graphql"

    def __call__(self, chembl_id: str) -> dict:
        query = (
            "query($cid: String!) {"
            " molecule(chemblId: $cid) {"
            "  canonicalSmiles"
            "  activities(first: 50) { edges { node { standardType standardValue standardUnits } } }"
            " }"
            "}"
        )
        try:
            resp = requests.post(
                self.endpoint,
                json={"query": query, "variables": {"cid": chembl_id}},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()["data"]["molecule"]
        except Exception as exc:  # pragma: no cover - network issues
            return {"error": str(exc)}

        smiles = data.get("canonicalSmiles")
        acts = [edge["node"] for edge in data.get("activities", {}).get("edges", [])]
        return {"smiles": smiles, "activities": acts}
