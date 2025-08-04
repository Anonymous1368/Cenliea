from rdflib import Graph
from collections import defaultdict
from Param import *
import time
import os
import sys

# Ensure the intermediate input directory exists
if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)
start_time = time.time()


if __name__ == '__main__':
    """
    Step 1 of the pipeline:
    - Converts source/target RDF KGs to N-Triples format
    - Parses the reference alignment file (TTL or XML)
    - Extracts aligned entity pairs
    - Saves output in INPUT_PATH/same_as
    """
    start_time = time.time()
    RAW_DATA_PATH = "./raw_files/"+DATASET+'/'
    OUTPUT_NAMES = [INPUT_PATH+"kg1_triples", INPUT_PATH+"kg2_triples"]
    not_blank_lines = [[],[]] # to hold non-blank triples

    # Process both source and target KG files
    for i in range(2):
        g = Graph()
        f = RAW_DATA_PATH+KG_FILES[i]
        g.parse(f, format=KG_FORMAT)
        fname = OUTPUT_NAMES[i]
        g.serialize(destination= fname, format="nt")

        # Lowercase all triples and remove newlines
        with open(fname, "r") as f:
            for line in f.readlines():
                line = line.rstrip('\n')
                not_blank_lines[i].append(line.lower())
        # Overwrite file with normalized lines
        with open(fname, 'w') as f:
            for nb in not_blank_lines[i]:
                f.write(nb+"\n")

    # Parse the reference alignment file to extract aligned entity pairs
    g = Graph()
    g.parse(RAW_DATA_PATH+ALIGN_FILE, format=ALIGN_FORMAT)
    fname = INPUT_PATH+"same_as"
    g.serialize(destination=fname+"_raw", format="nt")

    # TTL format: direct parsing of triple lines
    if ALIGN_FORMAT == 'ttl':
        with open(fname+"_raw", "r") as f:

            f_read_lines = f.readlines()
            pairs_list = []

            for line in f_read_lines:
                h, _, t = line.rstrip('\n').split(' ',2)
                h = h.strip('<>')
                t = t.strip('<>')
                t = t.strip('> .')
                pairs_list.append((t,h))

    # XML format: parse using predicates like alignmententity1 and alignmententity2
    elif ALIGN_FORMAT == 'xml':
        ent_pairs_1 = defaultdict(list)
        ent_pairs_2 = defaultdict(list)
        with open(fname+"_raw", "r") as f:
            for line in f.readlines():
                l = line.rstrip('\n').split(' ',2)
                t = l[-1]
                if t.startswith('<http://') and "alignmententity1" in l[1]:
                    t = t.strip('<>')
                    t = t.strip('> .')
                    ent_pairs_1[l[0]].append(t)
                if t.startswith('<http://') and "alignmententity2" in l[1]:
                    t = t.strip('<>')
                    t = t.strip('> .')
                    ent_pairs_2[l[0]].append(t)

        # Match up entity1 and entity2 pairs
        pairs_list = []
        for ent1 in ent_pairs_1.keys():
            if ent1 in ent_pairs_2.keys():
                for t in ent_pairs_2[ent1]:
                    pairs_list.append((ent_pairs_1[ent1][0], t))
                if len(ent_pairs_2[ent1])>1:
                    print("For {} More than 1 aligned entity existed!".format(ent_pairs_1[ent1]))
                if len(ent_pairs_1[ent1])>1:
                    print("More than 1 member existed in ent_pairs_1 for this entity: ", ent1)

    # Save the aligned entity pairs
    print("Number of aligned entities: ", len(pairs_list))
    with open(fname, 'w') as f:
        for pair in sorted(pairs_list):
            f.write(pair[0].lower()+" "+pair[1].lower()+'\n')
    # Remove intermediate raw file
    os.remove(fname+"_raw")
    print("--- Runtime: %s seconds ---" % (time.time() - start_time))
    # sys.stdout = orig_stdout
    # f.close()
