import sys
from types import SimpleNamespace
import importlib
import importlib.util


def get_requests_pkg():
    if "requests" in sys.modules:
        del sys.modules["requests"]
    path = sys.path.pop(0)
    try:
        spec = importlib.util.find_spec("requests")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.insert(0, path)


def import_bio_with_patch(entrez_mod):
    """Import tools.bio after injecting a fake Bio module."""
    sys.modules["Bio"] = SimpleNamespace(Entrez=entrez_mod)
    for mod in ["tools.bio", "tools.bio.pubmed", "tools.bio.chembl"]:
        sys.modules.pop(mod, None)
    importlib.invalidate_caches()
    import tools.bio  # noqa: F401

    return sys.modules["tools.bio"]


def test_pubmed_tool(monkeypatch):
    class DummyHandle:
        def __init__(self, data):
            self.data = data

        def close(self):
            pass

    search_handle = DummyHandle({"IdList": ["1", "2", "3"]})
    fetch_handle = DummyHandle(
        {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "1",
                        "Article": {
                            "ArticleTitle": "A",
                            "Abstract": {"AbstractText": ["alpha"]},
                        },
                    }
                },
                {
                    "MedlineCitation": {
                        "PMID": "2",
                        "Article": {
                            "ArticleTitle": "B",
                            "Abstract": {"AbstractText": ["beta"]},
                        },
                    }
                },
                {
                    "MedlineCitation": {
                        "PMID": "3",
                        "Article": {
                            "ArticleTitle": "C",
                            "Abstract": {"AbstractText": ["gamma"]},
                        },
                    }
                },
            ]
        }
    )

    ent = SimpleNamespace(
        esearch=lambda **kw: search_handle,
        efetch=lambda **kw: fetch_handle,
        read=lambda h: h.data,
    )
    bio = import_bio_with_patch(ent)
    res = bio.PubMedTool()("liver", top_k=3)
    assert len(res) >= 3
    assert res[0]["pmid"] == "1"
    assert res[0]["title"] == "A"
    assert res[0]["abstract"] == "alpha"


def test_chembl_tool(monkeypatch):
    payload = {
        "data": {
            "molecule": {
                "canonicalSmiles": "C",
                "activities": {
                    "edges": [
                        {
                            "node": {
                                "standardType": "IC50",
                                "standardValue": "50",
                                "standardUnits": "nM",
                            }
                        }
                    ]
                },
            }
        }
    }

    class FakeResp:
        def json(self):
            return payload

        def raise_for_status(self):
            pass

    dummy = SimpleNamespace(post=lambda *a, **kw: FakeResp())
    monkeypatch.setitem(sys.modules, "requests", dummy)

    bio = import_bio_with_patch(SimpleNamespace())
    res = bio.ChEMBLTool()("CHEMBL297")
    assert res["smiles"] == "C"
    assert res["activities"][0]["standardType"] == "IC50"


def test_vina_tool_returns_energy():
    bio = import_bio_with_patch(SimpleNamespace())
    receptor = "/usr/share/autodock/Tests/1pgp_rigid.pdbqt"
    ligand = "/usr/share/autodock/Tests/1pgp_lig.pdbqt"
    dg = bio.VinaDockingTool()(receptor, ligand)
    assert dg is not None
    assert dg <= 0


def test_qsar_tool_shape():
    bio = import_bio_with_patch(SimpleNamespace())
    probs = bio.QSARTool()("CCO")
    assert isinstance(probs, list)
    assert len(probs) == 12
