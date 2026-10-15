"""
Scan universe: sector ETFs, thematic ETFs and thematic stock baskets.

Every stock carries one GICS-style sector and one or more themes, so the
scanner can roll results up either way. Override or extend the built-in
universe with a CSV (columns: ticker, sector, themes) where ``themes`` is a
``|``-separated list, e.g. ``NVDA,Technology,AI Semis|Data Center``.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BENCHMARK = "SPY"

# SPDR sector ETFs -> sector name
SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
}

# Industry / thematic ETFs -> theme name
THEME_ETFS: Dict[str, str] = {
    "SMH": "AI Semis",
    "IGV": "Software / Cloud",
    "CIBR": "Cybersecurity",
    "BOTZ": "Robotics / Automation",
    "QTUM": "Quantum Computing",
    "URA": "Nuclear / Uranium",
    "TAN": "Solar / Clean Energy",
    "LIT": "Lithium / Batteries",
    "ITA": "Defense / Aerospace",
    "UFO": "Space",
    "XBI": "Biotech",
    "IHI": "Medical Devices",
    "BLOK": "Crypto / Blockchain",
    "KRE": "Regional Banks",
    "XHB": "Homebuilders",
    "XRT": "Retail",
    "GDX": "Gold Miners",
    "SIL": "Silver Miners",
    "COPX": "Copper Miners",
    "XOP": "Oil & Gas E&P",
    "JETS": "Airlines / Travel",
    "PAVE": "Infrastructure",
    "MSOS": "Cannabis",
    "IPAY": "Fintech / Payments",
}

# theme -> (sector, tickers)
THEMES: Dict[str, tuple] = {
    "AI Semis": ("Technology", [
        "NVDA", "AMD", "AVGO", "MRVL", "ARM", "MU", "TSM", "ALAB", "CRDO",
        "SMCI", "LSCC", "AMKR", "ONTO", "AMAT", "LRCX", "KLAC", "INTC", "QCOM",
    ]),
    "Data Center / AI Infra": ("Technology", [
        "VRT", "ANET", "DELL", "CLS", "COHR", "CIEN", "NBIS", "CRWV", "APLD",
        "IREN", "WULF", "ETN", "PWR", "GEV",
    ]),
    "Software / Cloud": ("Technology", [
        "PLTR", "SNOW", "DDOG", "MDB", "NET", "CRM", "NOW", "SHOP", "HUBS",
        "TEAM", "APP", "U", "PATH", "AI", "SOUN", "BBAI",
    ]),
    "Cybersecurity": ("Technology", [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RBRK", "TENB",
    ]),
    "Quantum Computing": ("Technology", [
        "IONQ", "RGTI", "QBTS", "QUBT", "ARQQ",
    ]),
    "Robotics / Automation": ("Industrials", [
        "ISRG", "TER", "ROK", "SYM", "SERV", "RR", "PDYN", "KITT",
    ]),
    "Nuclear / Uranium": ("Energy", [
        "CCJ", "UEC", "NXE", "DNN", "UUUU", "LEU", "OKLO", "SMR", "NNE", "BWXT",
    ]),
    "Solar / Clean Energy": ("Utilities", [
        "FSLR", "ENPH", "SEDG", "RUN", "NXT", "ARRY", "PLUG", "BE", "FCEL",
    ]),
    "Lithium / Batteries": ("Materials", [
        "ALB", "SQM", "LAC", "SLDP", "QS", "EOSE", "FLNC", "ENVX",
    ]),
    "Defense / Aerospace": ("Industrials", [
        "LMT", "NOC", "RTX", "GD", "LHX", "KTOS", "AVAV", "AXON", "HII", "MRCY",
    ]),
    "Space": ("Industrials", [
        "RKLB", "ASTS", "LUNR", "RDW", "PL", "SPCE", "BKSY", "SATS",
    ]),
    "Drones / eVTOL": ("Industrials", [
        "JOBY", "ACHR", "EH", "UMAC", "ONDS", "DPRO",
    ]),
    "Biotech": ("Health Care", [
        "VRTX", "REGN", "MRNA", "CRSP", "BEAM", "NTLA", "SRPT", "VKTX",
        "EXEL", "INSM", "ALNY", "RXRX", "ARWR", "IONS",
    ]),
    "GLP-1 / Obesity": ("Health Care", [
        "LLY", "NVO", "VKTX", "AMGN", "HIMS", "TERN",
    ]),
    "Medical Devices": ("Health Care", [
        "ISRG", "BSX", "SYK", "DXCM", "PODD", "TMDX", "GMED",
    ]),
    "Crypto / Blockchain": ("Financials", [
        "COIN", "MSTR", "HOOD", "MARA", "RIOT", "CLSK", "HUT", "BITF", "CIFR",
        "GLXY", "CRCL",
    ]),
    "Fintech / Payments": ("Financials", [
        "SOFI", "AFRM", "UPST", "PYPL", "XYZ", "TOST", "NU", "DAVE", "LC",
    ]),
    "Regional Banks": ("Financials", [
        "KRE", "ZION", "WAL", "CFG", "KEY", "FITB", "HBAN", "RF",
    ]),
    "Homebuilders": ("Consumer Discretionary", [
        "DHI", "LEN", "PHM", "TOL", "KBH", "NVR", "BLDR",
    ]),
    "Retail / Consumer": ("Consumer Discretionary", [
        "AMZN", "COST", "WMT", "TGT", "ANF", "DECK", "ONON", "CROX", "LULU",
        "ELF", "CAVA", "SG",
    ]),
    "EV / Autos": ("Consumer Discretionary", [
        "TSLA", "RIVN", "LCID", "NIO", "XPEV", "LI", "F", "GM",
    ]),
    "Gold & Silver Miners": ("Materials", [
        "NEM", "AEM", "GOLD", "KGC", "AU", "PAAS", "AG", "HL", "CDE", "EXK",
        "IAG", "EQX",
    ]),
    "Copper / Rare Earths": ("Materials", [
        "FCX", "SCCO", "TECK", "HBM", "MP", "USAR", "ERO",
    ]),
    "Oil & Gas": ("Energy", [
        "XOM", "CVX", "OXY", "DVN", "FANG", "EOG", "APA", "AR", "EQT", "RRC",
    ]),
    "Power / Utilities": ("Utilities", [
        "VST", "CEG", "NRG", "TLN", "NEE", "SO", "DUK",
    ]),
    "Communication / Media": ("Communication Services", [
        "META", "GOOGL", "NFLX", "RDDT", "SPOT", "PINS", "SNAP", "RBLX", "TTWO",
    ]),
    "Travel / Leisure": ("Consumer Discretionary", [
        "UBER", "ABNB", "BKNG", "DAL", "UAL", "AAL", "CCL", "RCL", "NCLH",
    ]),
    "Cannabis": ("Health Care", [
        "TLRY", "CGC", "ACB", "CRON", "MSOS",
    ]),
    "Staples": ("Consumer Staples", [
        "PG", "KO", "PEP", "MDLZ", "CL", "KHC", "CELH", "MNST",
    ]),
    "REITs / Real Estate": ("Real Estate", [
        "PLD", "AMT", "EQIX", "DLR", "O", "SPG", "WELL", "IRM",
    ]),
}


@dataclass
class Universe:
    """Ticker -> sector / themes mapping plus the ETF lists."""

    sectors: Dict[str, str] = field(default_factory=dict)
    themes: Dict[str, List[str]] = field(default_factory=dict)
    sector_etfs: Dict[str, str] = field(default_factory=lambda: dict(SECTOR_ETFS))
    theme_etfs: Dict[str, str] = field(default_factory=lambda: dict(THEME_ETFS))
    benchmark: str = BENCHMARK

    @property
    def stocks(self) -> List[str]:
        return sorted(self.sectors)

    @property
    def all_tickers(self) -> List[str]:
        return sorted(set(self.stocks) | set(self.sector_etfs)
                      | set(self.theme_etfs) | {self.benchmark})

    def theme_members(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for t, ths in self.themes.items():
            for th in ths:
                out.setdefault(th, []).append(t)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def sector_members(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for t, sec in self.sectors.items():
            out.setdefault(sec, []).append(t)
        return {k: sorted(v) for k, v in sorted(out.items())}

    def add(self, ticker: str, sector: str, themes: List[str]) -> None:
        ticker = ticker.strip().upper()
        self.sectors.setdefault(ticker, sector)
        existing = self.themes.setdefault(ticker, [])
        for th in themes:
            if th and th not in existing:
                existing.append(th)

    def filter_themes(self, wanted: Optional[List[str]]) -> "Universe":
        """Return a copy limited to stocks in any of ``wanted`` (case-insensitive substring)."""
        if not wanted:
            return self
        keys = [w.lower() for w in wanted]
        u = Universe(sector_etfs=self.sector_etfs, theme_etfs=self.theme_etfs,
                     benchmark=self.benchmark)
        for t, ths in self.themes.items():
            if any(k in th.lower() for th in ths for k in keys):
                u.add(t, self.sectors[t], ths)
        return u


def default_universe() -> Universe:
    u = Universe()
    for theme, (sector, tickers) in THEMES.items():
        for t in tickers:
            u.add(t, sector, [theme])
    return u


def load_universe_csv(path: str, base: Optional[Universe] = None) -> Universe:
    """Load (or merge into ``base``) a universe CSV with ticker,sector,themes columns."""
    u = base if base is not None else Universe()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                continue
            sector = (row.get("sector") or "Unclassified").strip()
            themes = [s.strip() for s in (row.get("themes") or "Custom").split("|")]
            u.add(ticker, sector, themes)
    return u
