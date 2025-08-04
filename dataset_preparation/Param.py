# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-d','--DatasetFolderName', help='dataset folder name located itself in raw_files folder. Dataset folder name should not contain any punctuations.', required=True)

#Info about the source, target and reference alignment files
KG_FILES = ["source.xml", "target.xml"] #source and target kg file names
KG_FORMAT = KG_FILES[0].split('.')[-1] #Format of the input file, could be ttl, xml, nt. #NO need to change

# args = vars(parser.parse_args())
DATASET =  'Doremus' #'2023StarwarsSWTOR' #'IM2022SPIMBENCHsmall' 2023MemoryalphaStexpanded 2023MemoryalphaMemorybeta  2023marvelcinematicuniverseMarvel 2023StarwarsSWG AgroLD Doremus #args['DatasetFolderName'] #Attention: Dataset folder name should not contain any punctuations

ALIGN_FILE = "reference.xml" #Name of the file containing reference alignment e.g. same_as.ttl or refDHT.rdf "refDHT.ttl"
ALIGN_FORMAT =ALIGN_FILE.split('.')[-1] #Format of the reference alignment file, could be ttl, xml or nt. #NO need to change
MAX_ATTR_VAL_LEN = 48 #64 max length (max number of characters) of attribute values
MAX_ATTR_NO = 5 #Max number of attributes considered for each entity
MAX_REL_NO = 3 #MAx number of entitity's neighbors that their attribute values added to the info of the entity
MAX_CANDID = 1000 #Max number of cnadidates randomly selected as negative pairs if the KG is large
INPUT_PATH = './EALLM_inputs/'+DATASET+'/'
ALL_DATA_PATH = './EALLM_inputs/'
FINAL_INPUT = './final_input/'
RAW_DATA_PATH = './raw_files/'
NEG_SAMPLES = 1 #Set the number of negative samples to be generated for test set
BATCH_SIZE = 2
TEST = 0.3 # 0.3 : 30% of the data would be included as the test set
SHARD_SIZE = 7500 #Number of samples in each train data shard 7500 , 10000, etc.
MIXED_DATASET_PATH = "./Mixed"     #PATH TO FINAL DATASET CONTAINING ALL PROMPTS EXCEPT FOR THE INSTRUCTION (SIM&DIFF)
MIXED_DATA_CONFIG = False  #False True 
