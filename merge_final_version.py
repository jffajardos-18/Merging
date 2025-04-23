###############################################################################
# 0) Imports & config
###############################################################################
import pandas as pd
import numpy as np
import re, unicodedata, uuid, time
from urllib.parse import urlparse
from itertools import combinations
from rapidfuzz import fuzz
import jellyfish

try:
    import tldextract
except ImportError:
    tldextract = None

# ----- paths -----------------------------------------------------------------
file1_path  = r"C:\Users\.xlsx"
file2_path  = r"C:\Users\.xlsx"
output_path = r"C:\Users\.xlsx"

# ----- tunables --------------------------------------------------------------
PRIORITY          = {"file1": 0, "file2": 1}
MIN_CLUSTER_SCORE = 80           # flag cluster if any pair drops below this

SUFFIX_RE = re.compile(
    r"\b(inc(orporated)?|corp(oration)?|co(mpany)?|ltd|llc|plc|gmbh|s\.a\.s|"
    r"s\.a\.|oy|spa|pte|sa|bv|kg|ag|kft)\b",
    flags=re.I,
)
PUNCT_RE = re.compile(r"[,&.\-]")


def tstamp() -> str:
    """Return HH:MM:SS for quick visual timing."""
    return time.strftime("%H:%M:%S")


###############################################################################
# 1) Normalisers
###############################################################################
def normalize_company_name(name: str) -> str:
    if pd.isna(name):
        return ""
    txt = unicodedata.normalize("NFKD", str(name))
    txt = txt.encode("ascii", "ignore").decode("ascii").lower()
    txt = PUNCT_RE.sub("", txt)
    txt = SUFFIX_RE.sub("", txt)
    return re.sub(r"\s+", "", txt).strip()


def normalize_website(url: str) -> str:
    if pd.isna(url) or str(url).strip() == "":
        return ""
    u = str(url).strip().lower()
    if not u.startswith(("http://", "https://")):
        u = "http://" + u
    if tldextract:
        ext = tldextract.extract(u)
        return ext.domain + (f".{ext.suffix}" if ext.suffix else "")
    parsed = urlparse(u)
    dom = (parsed.netloc or parsed.path).lstrip("www.")
    return dom.split(":")[0]


###############################################################################
# 2) Pairwise comparator  (website Soundex removed)
###############################################################################
def compare_pair(uid_i, uid_j, rec_i, rec_j):
    s_name = fuzz.ratio(rec_i["norm_company_name"], rec_j["norm_company_name"])
    s_web = fuzz.ratio(rec_i["norm_website"], rec_j["norm_website"])
    comb_i = rec_i["norm_company_name"] + rec_i["norm_website"]
    comb_j = rec_j["norm_company_name"] + rec_j["norm_website"]
    s_comb = fuzz.ratio(comb_i, comb_j)

    m_name = rec_i["soundex_name"] == rec_j["soundex_name"]
    m_comb = rec_i["soundex_key"] == rec_j["soundex_key"]
    thresh = {2: 85, 1: 90, 0: 95}[m_name + m_comb]

    if s_comb >= thresh:
        keep_uid = (
            uid_i
            if PRIORITY[rec_i["source_file"]] <= PRIORITY[rec_j["source_file"]]
            else uid_j
        )
        drop_uid = uid_j if keep_uid == uid_i else uid_i
        reason = (
            f"Fuzzy match (name={s_name}, web={s_web}, "
            f"comb={s_comb}, soundex_key={rec_i['soundex_key']})"
        )
        return keep_uid, drop_uid, reason, s_comb
    return None


###############################################################################
# 3) Read & enrich
###############################################################################
print(f"[{tstamp()}] Reading Excel files …")
df1, df2 = pd.read_excel(file1_path), pd.read_excel(file2_path)
print(f"[{tstamp()}]  » file1 rows={len(df1):,}, file2 rows={len(df2):,}")

for df, tag in ((df1, "file1"), (df2, "file2")):
    df["uid"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["source_file"] = tag
    df["Company name deduplicada"] = df["Company name"]

print(f"[{tstamp()}] Normalising …")
for df in (df1, df2):
    df["norm_company_name"] = df["Company name"].apply(normalize_company_name)
    df["norm_website"] = df["Company website"].apply(normalize_website)

combined = pd.concat([df1, df2], ignore_index=True, sort=False)
print(f"[{tstamp()}] Combined rows={len(combined):,}")

###############################################################################
# 4) Phonetic keys
###############################################################################
combined["soundex_name"] = combined["norm_company_name"].apply(jellyfish.soundex)
combined["soundex_key"] = combined.apply(
    lambda r: jellyfish.soundex(
        r["norm_company_name"]
        + ("." + r["norm_website"] if r["norm_website"] else "")
    ),
    axis=1,
)
print(f"[{tstamp()}] Phonetic keys ready")

###############################################################################
# 5) Duplicate detection
###############################################################################
removal_log, exact_pairs, seen_pairs = [], [], set()
uid_to_idx = dict(zip(combined["uid"], combined.index))


def log_link(uid_a, uid_b, reason, score=None):
    pair = tuple(sorted((uid_a, uid_b)))
    if pair in seen_pairs:
        return
    seen_pairs.add(pair)
    a, b = uid_to_idx[uid_a], uid_to_idx[uid_b]
    removal_log.append(
        {
            "file1_name": combined.at[a, "Company name"],
            "file2_name": combined.at[b, "Company name"],
            "file1_website": combined.at[a, "Company website"],
            "file2_website": combined.at[b, "Company website"],
            "Reason": reason,
            "Matching Score": score,
            "soundex_key": combined.at[a, "soundex_key"],
        }
    )
    exact_pairs.append(pair)


print(f"[{tstamp()}] –– Exact duplicates by name …")
for name, grp in combined.groupby("norm_company_name"):
    if name and len(grp) > 1:
        keep_uid = grp.loc[grp["source_file"].map(PRIORITY).idxmin(), "uid"]
        for uid in grp["uid"]:
            if uid != keep_uid:
                log_link(keep_uid, uid, "Exact match on name")

print(f"[{tstamp()}] –– Exact duplicates by website …")
for site, grp in combined.groupby("norm_website"):
    if site and len(grp) > 1:
        keep_uid = grp.loc[grp["source_file"].map(PRIORITY).idxmin(), "uid"]
        for uid in grp["uid"]:
            if uid != keep_uid:
                log_link(keep_uid, uid, "Exact match on website")

print(f"[{tstamp()}] Exact‑match pairs logged: {len(removal_log):,}")

# --- fuzzy pass --------------------------------------------------------------
pairs = {
    tuple(sorted(p))
    for _, g in combined.groupby("soundex_key")
    for p in combinations(g["uid"], 2)
}

total_pairs = len(pairs)
print(f"[{tstamp()}] Fuzzy‑check candidate pairs: {total_pairs:,}")

for n, (uid_i, uid_j) in enumerate(pairs, 1):
    if n % 200_000 == 0 or n == total_pairs:
        print(f"[{tstamp()}]   fuzzy compared {n:,}/{total_pairs:,}")

    res = compare_pair(
        uid_i,
        uid_j,
        combined.loc[uid_to_idx[uid_i]].to_dict(),
        combined.loc[uid_to_idx[uid_j]].to_dict(),
    )
    if res:
        keep_uid, drop_uid, reason, score = res
        log_link(keep_uid, drop_uid, reason, score)

print(
    f"[{tstamp()}] Total duplicate links (after de‑duping): {len(exact_pairs):,}"
)

###############################################################################
# 6) Union‑find clustering
###############################################################################
class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            p = self.find(p)
            self.parent[x] = p
        return p

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


uf = UnionFind(combined["uid"])
for uid_a, uid_b in exact_pairs:
    uf.union(uid_a, uid_b)
combined["group_id"] = combined["uid"].apply(uf.find)

print(
    f"[{tstamp()}] Clustered into {combined['group_id'].nunique():,} groups"
)

###############################################################################
# 7) Cluster QA
###############################################################################
flagged = []
for root, grp in combined.groupby("group_id"):
    if len(grp) < 2:
        continue
    scores = [
        fuzz.ratio(
            a["norm_company_name"] + a["norm_website"],
            b["norm_company_name"] + b["norm_website"],
        )
        for a, b in combinations(grp.to_dict("records"), 2)
        if (a["norm_company_name"] or a["norm_website"])
        and (b["norm_company_name"] or b["norm_website"])
    ]
    if scores and min(scores) < MIN_CLUSTER_SCORE:
        flagged.append(
            {
                "group_id": root,
                "uids": "; ".join(grp["uid"]),
                "min_combined_score": min(scores),
            }
        )

print(f"[{tstamp()}] Flagged questionable clusters: {len(flagged):,}")

###############################################################################
# 8) Merge each cluster (no row dropped)
###############################################################################
final_columns = sorted(set(df1.columns) | set(df2.columns)) + [
    "Company name deduplicada"
]


def merge_group(records):
    merged = {}
    file1_vals = {r["uid"]: r for r in records if r["source_file"] == "file1"}

    for col in final_columns:
        vals = []
        for r in records:
            v = r.get(col)
            if (
                v not in (None, "", np.nan)
                and not pd.isna(v)
                and v not in vals
            ):
                vals.append(v)
        if not vals:
            merged[col] = None
            continue
        prim = next(
            (
                file1_vals[r["uid"]][col]
                for r in records
                if r["uid"] in file1_vals and r.get(col) in vals
            ),
            vals[0],
        )
        merged[col] = prim
        others = [v for v in vals if v != prim]
        if others:
            merged[f"{col}_conflict"] = "; ".join(map(str, others))
    merged["source_file"] = (
        "file1"
        if any(r["source_file"] == "file1" for r in records)
        else "file2"
    )
    return merged


print(f"[{tstamp()}] Merging clusters …")
merged_records, unique_file1_records, unique_file2_records = [], [], []
for root, grp in combined.groupby("group_id"):
    recs = grp.to_dict("records")
    merged = merge_group(recs) if len(recs) > 1 else recs[0]
    merged["group_id"] = root
    merged_records.append(merged)
    sources = set(grp["source_file"])
    if sources == {"file1"}:
        unique_file1_records.append(merged)
    elif sources == {"file2"}:
        unique_file2_records.append(merged)

print(f"[{tstamp()}]  » final merged records: {len(merged_records):,}")
print(f"[{tstamp()}]  » unique only‑file1:    {len(unique_file1_records):,}")
print(f"[{tstamp()}]  » unique only‑file2:    {len(unique_file2_records):,}")

###############################################################################
# 9) DataFrames & Excel
###############################################################################
df_merged      = pd.DataFrame(merged_records)
df_removed_log = pd.DataFrame(removal_log)
df_unique1     = pd.DataFrame(unique_file1_records)
df_unique2     = pd.DataFrame(unique_file2_records)
df_flagged     = pd.DataFrame(flagged)

helpers = ["norm_company_name", "norm_website",
           "soundex_name", "soundex_key", "uid"]

df_merged.drop(columns=[c for c in helpers if c in df_merged.columns],
               inplace=True, errors="ignore")

print(f"[{tstamp()}] Writing Excel …")

# — use openpyxl (streams correctly) —
with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    df_merged.to_excel(writer, sheet_name="Clean Data",        index=False)
    df_removed_log.to_excel(writer, sheet_name="Removed Duplicates", index=False)
    df_unique1.to_excel(writer,   sheet_name="Unique File1",   index=False)
    df_unique2.to_excel(writer,   sheet_name="Unique File2",   index=False)
    if not df_flagged.empty:
        df_flagged.to_excel(writer, sheet_name="Flagged Clusters", index=False)

print(f"[{tstamp()}] Done. Results at: {output_path}")
