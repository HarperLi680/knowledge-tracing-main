import os
import pandas as pd
import numpy as np
import json

# Constants used in processing
ALL_KEYS = ["fold", "uid", "questions", "concepts", "responses", "timestamps",
            "usetimes", "selectmasks", "is_repeat", "qidxs", "rest", "orirow", "cidxs"]
ONE_KEYS = ["fold", "uid"]


def read_data(fname, min_seq_len=3, response_set=[0, 1]):
    """
    Reads data from a formatted text file and parses it into a dictionary.
    
    Args:
        fname: Path to input file
        min_seq_len: Minimum sequence length to keep
        response_set: Valid response values
        
    Returns:
        Tuple of (DataFrame, set of effective_keys)
    """
    effective_keys = set()
    dres = dict()
    delstu, delnum, badr = 0, 0, 0
    goodnum = 0
    
    with open(fname, "r", encoding="utf8") as fin:
        i = 0
        lines = fin.readlines()
        dcur = dict()
        while i < len(lines):
            line = lines[i].strip()
            if i % 6 == 0:  # stuid
                effective_keys.add("uid")
                tmps = line.split(",")
                if "(" in tmps[0]:
                    stuid, seq_len = tmps[0].replace('(', ''), int(tmps[2])
                else:
                    stuid, seq_len = tmps[0], int(tmps[1])
                if seq_len < min_seq_len:
                    i += 6
                    dcur = dict()
                    delstu += 1
                    continue
                dcur["uid"] = stuid
            elif i % 6 == 1:  # questions
                qs = []
                if line.find("NA") == -1:
                    effective_keys.add("questions")
                    qs = line.split(",")
                dcur["questions"] = qs
            elif i % 6 == 2:  # concepts
                cs = []
                if line.find("NA") == -1:
                    effective_keys.add("concepts")
                    cs = line.split(",")
                dcur["concepts"] = cs
            elif i % 6 == 3:  # responses
                rs = []
                flag = True
                if line.find("NA") == -1:
                    effective_keys.add("responses")
                    for r in line.split(","):
                        try:
                            r = int(r)
                            if r not in response_set:
                                print(f"error response in line: {i}")
                                flag = False
                                break
                            rs.append(r)
                        except:
                            print(f"error response in line: {i}")
                            flag = False
                            break
                    if not flag:
                        i += 3
                        dcur = dict()
                        badr += 1
                        continue
                dcur["responses"] = rs
            elif i % 6 == 4:  # timestamps
                ts = []
                if line.find("NA") == -1:
                    effective_keys.add("timestamps")
                    ts = line.split(",")
                dcur["timestamps"] = ts
            elif i % 6 == 5:  # usetimes
                usets = []
                if line.find("NA") == -1:
                    effective_keys.add("usetimes")
                    usets = line.split(",")
                dcur["usetimes"] = usets
                
                # Validation and saving
                if len(dcur["responses"]) < min_seq_len:
                    i += 1
                    dcur = dict()
                    delnum += len(dcur["responses"])
                    continue
                
                goodnum += 1
                for key in effective_keys:
                    dres.setdefault(key, [])
                    if key != "uid":
                        dres[key].append(",".join([str(k) for k in dcur[key]]))
                    else:
                        dres[key].append(dcur[key])
                dcur = dict()
            i += 1

    df = pd.DataFrame(dres)
    print(f"delete bad stu num of len: {delstu}, delete interactions: {delnum}, of r: {badr}, good num: {goodnum}")
    return df, effective_keys


def id_mapping(df):
    """
    Maps string IDs (questions, concepts, uids) to continuous integers.
    
    Args:
        df: Input dataframe
        
    Returns:
        Tuple of (mapped_df, id_mapping_dict)
    """
    id_keys = ["questions", "concepts", "uid"]
    dres = dict()
    dkeyid2idx = dict()
    
    print(f"df.columns: {df.columns}")
    for key in df.columns:
        if key not in id_keys:
            dres[key] = df[key]
            
    for i, row in df.iterrows():
        for key in id_keys:
            if key not in df.columns:
                continue
            dkeyid2idx.setdefault(key, dict())
            dres.setdefault(key, [])
            curids = []
            for id in row[key].split(","):
                if id not in dkeyid2idx[key]:
                    dkeyid2idx[key][id] = len(dkeyid2idx[key])
                curids.append(str(dkeyid2idx[key][id]))
            dres[key].append(",".join(curids))
            
    finaldf = pd.DataFrame(dres)
    return finaldf, dkeyid2idx


def generate_sequences(df, effective_keys, min_seq_len=3, maxlen=200, pad_val=-1):
    """
    Generates fixed-length sequences using sliding window or splitting for training.
    
    Args:
        df: Input dataframe
        effective_keys: Set of keys to process
        min_seq_len: Minimum sequence length
        maxlen: Maximum sequence length
        pad_val: Padding value
        
    Returns:
        DataFrame with sequences
    """
    save_keys = list(effective_keys) + ["selectmasks"]
    dres = {"selectmasks": []}
    dropnum = 0
    
    for i, row in df.iterrows():
        dcur = {}
        for key in effective_keys:
            if key not in ONE_KEYS:
                dcur[key] = row[key].split(",")
            else:
                dcur[key] = row[key]

        rest, lenrs = len(dcur["responses"]), len(dcur["responses"])
        j = 0
        while lenrs >= j + maxlen:
            rest = rest - maxlen
            for key in effective_keys:
                dres.setdefault(key, [])
                if key not in ONE_KEYS:
                    dres[key].append(",".join(dcur[key][j: j + maxlen]))
                else:
                    dres[key].append(dcur[key])
            dres["selectmasks"].append(",".join(["1"] * maxlen))
            j += maxlen

        if rest < min_seq_len:
            dropnum += rest
            continue

        pad_dim = maxlen - rest
        for key in effective_keys:
            dres.setdefault(key, [])
            if key not in ONE_KEYS:
                paded_info = np.concatenate([dcur[key][j:], np.array([str(pad_val)] * pad_dim)])
                dres[key].append(",".join([str(k) for k in paded_info]))
            else:
                dres[key].append(dcur[key])
        dres["selectmasks"].append(",".join(["1"] * rest + [str(pad_val)] * pad_dim))

    dfinal = dict()
    for key in ALL_KEYS:
        if key in save_keys:
            dfinal[key] = dres[key]
    finaldf = pd.DataFrame(dfinal)
    print(f"dropnum: {dropnum}")
    return finaldf


def save_id2idx(dkeyid2idx, path):
    """Save ID mapping dictionary to JSON file."""
    with open(path, "w") as f:
        json.dump(dkeyid2idx, f)


def get_max_concepts(df):
    """Calculate maximum number of concepts per question."""
    max_concepts = 0
    if "concepts" in df.columns:
        for concepts in df["concepts"]:
            concept_list = concepts.split(",")
            for concept in concept_list:
                num_concepts = len(concept.split("_"))
                max_concepts = max(max_concepts, num_concepts)
    return max_concepts


def extend_multi_concepts(df, effective_keys):
    """
    Extend multi-concept entries in the dataframe.
    
    When a question is tagged with multiple concepts (e.g., "1_13"),
    this function splits it into separate interactions, one per concept.
    
    Args:
        df: Input dataframe with comma-separated sequences
        effective_keys: Set of column names to process
        
    Returns:
        Tuple of (extended_df, updated_effective_keys with 'is_repeat' added)
        
    Example:
        Input row:  uid=1, questions="q1,q2", concepts="c1,c2_c3", responses="0,1"
        Output row: uid=1, questions="q1,q2,q2", concepts="c1,c2,c3", responses="0,1,1", is_repeat="0,0,1"
    """
    # Early exit if no questions or concepts to extend
    if "questions" not in effective_keys or "concepts" not in effective_keys:
        print("No questions or concepts found - returning original dataframe")
        return df, effective_keys
    
    # Get all columns except uid (uid stays the same for each user)
    extend_keys = set(df.columns) - {"uid"}
    
    # Initialize result dictionary with uid column
    result = {"uid": df["uid"].tolist()}
    
    # Process each user's sequence
    for _, row in df.iterrows():
        # Parse all columns from comma-separated strings to lists
        parsed_row = {}
        for key in extend_keys:
            parsed_row[key] = row[key].split(",")
        
        # Build extended sequence for this user
        extended_sequence = {}
        
        # Process each interaction (question) in the sequence
        for i in range(len(parsed_row["questions"])):
            # Check if this interaction has multiple concepts (contains "_")
            concept_str = parsed_row["concepts"][i]
            
            if "_" in concept_str:
                # Multi-concept: split and create multiple interactions
                concept_ids = concept_str.split("_")
                
                # Add each concept separately
                extended_sequence.setdefault("concepts", [])
                extended_sequence["concepts"].extend(concept_ids)
                
                # Duplicate all other fields for each concept
                for key in extend_keys:
                    if key != "concepts":
                        extended_sequence.setdefault(key, [])
                        # Repeat this value len(concept_ids) times
                        extended_sequence[key].extend([parsed_row[key][i]] * len(concept_ids))
                
                # Mark first as original (0), rest as repeats (1)
                extended_sequence.setdefault("is_repeat", [])
                extended_sequence["is_repeat"].extend(["0"] + ["1"] * (len(concept_ids) - 1))
            
            else:
                # Single concept: just append normally
                for key in extend_keys:
                    extended_sequence.setdefault(key, [])
                    extended_sequence[key].append(parsed_row[key][i])
                
                # Mark as not a repeat
                extended_sequence.setdefault("is_repeat", [])
                extended_sequence["is_repeat"].append("0")
        
        # Convert lists back to comma-separated strings and add to result
        for key in extended_sequence:
            result.setdefault(key, [])
            result[key].append(",".join(extended_sequence[key]))
    
    # Create DataFrame from result
    final_df = pd.DataFrame(result)
    
    # Add is_repeat to effective keys
    effective_keys = set(effective_keys)  # Make a copy
    effective_keys.add("is_repeat")
    
    return final_df, effective_keys


def calStatistics(df, stares, key):
    """Calculate statistics for the dataset."""
    allin, allselect = 0, 0
    allqs, allcs = set(), set()
    
    for i, row in df.iterrows():
        rs = row["responses"].split(",")
        curlen = len(rs) - rs.count("-1")
        allin += curlen
        
        if "selectmasks" in row:
            ss = row["selectmasks"].split(",")
            slen = ss.count("1")
            allselect += slen
            
        if "concepts" in row:
            cs = row["concepts"].split(",")
            fc = []
            for c in cs:
                cc = c.split("_")
                fc.extend(cc)
            curcs = set(fc) - {"-1"}
            allcs |= curcs
            
        if "questions" in row:
            qs = row["questions"].split(",")
            curqs = set(qs) - {"-1"}
            allqs |= curqs
            
    stares.append(",".join([str(s) for s in [key, allin, df.shape[0], allselect]]))
    return allin, allselect, len(allqs), len(allcs), df.shape[0]