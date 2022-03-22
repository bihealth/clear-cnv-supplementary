# simulate sample
#%%

import subprocess, shlex, pathlib, tempfile,random, os, pickle
import pandas as pd, numpy as np
from io import StringIO
import multiprocessing as mp
import argparse
from logzero import logger,logfile
from tqdm import tqdm

#%%
 
# extract probes from reference, align them
def prepare_probes(path_reference, path_panel_bed, path_probes_bed, path_outprobes, cores, silent=False,**kwargs):
    tmpdir = tempfile.TemporaryDirectory()
    probes_bed = pathlib.Path(tmpdir.name) / "probes.bed"
    with open(probes_bed.name,'wb') as f:
        cmd_intersect = shlex.split(f"bedtools intersect -wa -a {path_probes_bed} -b {path_panel_bed}")
        subprocess.check_call(cmd_intersect,stdout=f)

    file_probes_fasta = pathlib.Path(tmpdir.name) / 'probes.fasta' 
    # bed file + ref to fasta
    cmd_probes_to_fasta = shlex.split(f"bedtools getfasta -fi {path_reference} -bed {probes_bed.name} -fo {file_probes_fasta}")
    if not silent:
        logger.info(f"extracting probes from reference:\n{' '.join(cmd_probes_to_fasta)}")
    p_probes_to_fasta   = subprocess.check_call(cmd_probes_to_fasta)
    # index fasta
    cmd_index_fasta = shlex.split(f"samtools faidx {file_probes_fasta}")
    if not silent:
        logger.info(f"indexing probes.fasta:\n{' '.join(cmd_index_fasta)}")
    subprocess.check_call(cmd_index_fasta)
    # align probes
    cmd_align_probes  = shlex.split(f"minimap2 -a -x sr -t {cores} {path_reference} {file_probes_fasta}")
    cmd_view_alprobes = shlex.split(f"samtools view -b")
    cmd_sort_alprobes = shlex.split(f"samtools sort -o {path_outprobes}")
    cmd_index_probes  = shlex.split(f"samtools index {path_outprobes}")
    if not silent:
        logger.info(f"align probes:\n{' '.join(cmd_align_probes)} | {' '.join(cmd_view_alprobes)} | {' '.join(cmd_sort_alprobes)}")
    p_align_probes  = subprocess.Popen(cmd_align_probes, stdout=subprocess.PIPE)
    p_view_alprobes = subprocess.Popen(cmd_view_alprobes, stdin=p_align_probes.stdout, stdout=subprocess.PIPE)
    p_sort_alprobes = subprocess.Popen(cmd_sort_alprobes, stdin= p_view_alprobes.stdout)
    p_sort_alprobes.communicate()
    subprocess.check_call(cmd_index_probes)
    if not silent:
        logger.info(f"done. Output written to: {path_outprobes}")

def get_cnvs(df,cnvs=(0,0),with_freq=False,min_exon=0,max_exon=0,seed=0):
    if sum(cnvs) < 1:
        raise Exception("cnvs must be set to a larger value than (0,0)")
    c = ["DEL","DUP"]
    chosen_cnvs = []
    for i,count in enumerate(cnvs):
        mask = ~np.zeros(df.shape[0],dtype=bool)
        mask &= df.loc[:,"CNV"] == c[i]
        if min_exon > 0:
            mask &= df["exons"] >= min_exon
        if max_exon > 0 and max_exon >= min_exon:
            mask &= df["exons"] <= max_exon
        dfs = df.loc[mask,:].copy()
        if dfs["freq"].sum() <= 0:
            dfs["freq"] = 1/dfs.shape[0]
        if count > dfs.shape[0]:
            logger.warn(f"Not enough CNVs to sample from. Picked all {str(dfs.shape[0])} out of desired {str(count)}. Seed={str(seed)}")
        x = dfs.sample(weights=dfs["freq"] if with_freq else None,n=min((count,dfs.shape[0])),replace=False,random_state=seed)
        chosen_cnvs.append(x)
    return pd.concat(chosen_cnvs)

def get_cnvs_of_order(workdir, panel,seed=0,cnvs=(0,0),with_freq=False,min_exons=0,max_exons=0,hom=False):
    path = pathlib.Path(pathlib.Path(workdir) / 'CNVs_dicts' / f'{panel}.counts.tsv')
    df = pd.read_csv(path,sep='\t',header = None)
    df.columns = ["chr","start","end","CNV","freq_het_alt","freq_hom_alt","exons"]
    if df.shape[0] == 0 or sum(cnvs) < 1:
        return pd.DataFrame(columns=df.columns)
    if sum(df["freq_hom_alt" if hom else "freq_het_alt"]) > 0:
        df["freq"] = df["freq_hom_alt" if hom else "freq_het_alt"] / sum(df["freq_hom_alt" if hom else "freq_het_alt"])
    return get_cnvs(df,cnvs,with_freq,min_exons,max_exons,seed)

#%%

def haplotype_seq(path_fastq_stem,path_wdir,path_reference,path_probes,seed,n_fragments,p_fmedian=500,p_illen=300):
    cmd_capsim = shlex.split(f"singularity exec \
        --bind {str(path_wdir)}:/{path_wdir.stem} \
        {'japsa/japsa.simg'} jsa.sim.capsim \
        --reference /{path_wdir.stem}/{str(os.path.relpath(path_reference,path_wdir))} \
        --probe /{path_wdir.stem}/{str(os.path.relpath(path_probes,path_wdir))} \
        --ID {seed} \
        --fmedian {p_fmedian} \
        --miseq /{path_wdir.stem}/{str(os.path.relpath(path_fastq_stem,path_wdir))} \
        --illen {p_illen} \
        --num {str(int(n_fragments))} \
        --ilmode=pe \
        --seed={seed} \
        --logFile=/{path_wdir.stem}/{str(os.path.relpath(path_wdir / 'logs' / str(seed),path_wdir))}")
    logger.info(f"generating reads on haplotype:\n{' '.join(cmd_capsim)}")
    subprocess.check_call(cmd_capsim)

def haplotype_seq_mp(
                        haplotypes:tuple,
                        path_wdir:pathlib.PosixPath,
                        path_reference:pathlib.PosixPath,
                        path_probes:pathlib.PosixPath,
                        n_fragments:int,
                        cores:int=4):
    pool = mp.Pool(cores)
    for i in range(len(haplotypes)):
        pool.apply_async(haplotype_seq, args=(haplotypes[i][2],path_wdir,path_reference,path_probes,haplotypes[i][3],n_fragments))
    pool.close()
    pool.join()

def align_pe_reads_to_ref(path_ref,paths_reads:list,path_bam,cores:int):
    logger.info(f"aligning reads on reference:\n{' '.join(list(map(str,paths_reads)))} on {path_ref} and write to {path_bam}")
    cmd_align = shlex.split(f"minimap2 -a -x sr -t {cores} {path_ref} {' '.join(list(map(str,paths_reads)))}")
    cmd_view  = shlex.split("samtools view -b")
    cmd_sort  = shlex.split(f"samtools sort -o {path_bam}")
    cmd_index = shlex.split(f"samtools index {path_bam}")
    p_align = subprocess.Popen(cmd_align,stdout=subprocess.PIPE)
    p_view  = subprocess.Popen(cmd_view, stdin=p_align.stdout,stdout=subprocess.PIPE)
    p_sort  = subprocess.Popen(cmd_sort, stdin=p_view.stdout)
    p_sort.communicate()
    subprocess.check_call(cmd_index)
    logger.info(f"finished alignment and written to {path_bam}.")

# cnvs_df get a table of only dels or dups to filter aligned reads accordingly to generate the CNV
def filter_bam(bam_file, cnvs_df, outbam):
    if len(set(cnvs_df["CNV"])) > 1:
        raise Exception("filter_bam(): argument 'cnvs_df' can only contain 'DEL' or 'DUP' entries")
    CN = set(cnvs_df["CNV"]).pop()
    regions_file = tempfile.NamedTemporaryFile(suffix='.bed')
    cnvs_df.to_csv(regions_file.name,sep='\t',header=False,index=False)
    outarg = "U" if CN == 'DEL' else "o"
    cmd = shlex.split(f"samtools view -b -L {regions_file.name} -{outarg} {outbam} {bam_file}")
    logger.info(' '.join(cmd))
    subprocess.check_call(cmd,stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def generate_true_cnv(
        workdir:pathlib.PosixPath,
        panel:str,
        seed:int,
        dels:int=0,
        dups:int=0,
        with_freq:bool=False,
        min_exons:int=1,
        max_exons:int=0,
        hom:bool=False,
        **kwargs):
    df_cnvs = get_cnvs_of_order(
                            workdir=workdir,
                            panel=panel,
                            seed=seed,
                            cnvs=(dels,dups),
                            with_freq=with_freq,
                            min_exons=min_exons,
                            max_exons=max_exons,
                            hom=hom)
    retdf = df_cnvs.copy()
    df_cnvs.loc[:,"CNV"] = ['HOM_'+cnv if hom else cnv for cnv in df_cnvs["CNV"]]
    df_cnvs.to_csv(pathlib.Path(workdir) / "true_cnvs" / ("cnvs_"+str(seed)+".tsv"),sep='\t',index=False)
    return retdf,df_cnvs

def sample_gen(
        workdir:pathlib.PosixPath,
        panel:str,
        seed:int,
        bam_path:pathlib.PosixPath,
        fasta_reference:pathlib.PosixPath,
        aligned_probes:pathlib.PosixPath,
        arr_fragments:pathlib.PosixPath,
        nfragments:int=0, # if 0 -> pick randomly
        dels:int=0,
        dups:int=0,
        with_freq:bool=False,
        min_exons:int=1,
        max_exons:int=0,
        hom:bool=False,
        cores:int=4,
        **kwargs):
    # pick num fragments
    logfile(pathlib.Path(workdir) / "log" / '.'.join(list(map(str,[panel,seed,dels,dups,'log']))))
    with open(arr_fragments,'rb') as f:
        arr = pickle.load(f)
    nfragments = random.choice(arr[panel]) if nfragments <= 0 else nfragments
    logger.info(f"approximating {str(nfragments)} fragments in total.")
    # pick CNVs
    df_cnvs = generate_true_cnv(
                workdir=workdir,
                panel=panel,
                seed=seed,
                dels=dels,
                dups=dups,
                with_freq=with_freq,
                min_exons=min_exons,
                max_exons=max_exons,
                hom=hom)
    # generate haplotypes
    #dir_tmp_fq = tempfile.TemporaryDirectory()
    #dir_fq = pathlib.Path(dir_tmp_fq.name)
    dir_fq = pathlib.Path(workdir) / "fastq"
    haplotypes = [
            ("h0",
                df_cnvs[df_cnvs["CNV"] == "DEL"].copy() if "DEL" in set(df_cnvs['CNV']) and hom else pd.DataFrame(),
                dir_fq / f"h0_{str(seed+0)}",
                seed+0),
            ("h1",
                df_cnvs[df_cnvs["CNV"] == "DEL"].copy() if "DEL" in set(df_cnvs['CNV']) else pd.DataFrame(),
                dir_fq / f"h1_{str(seed+1)}",
                seed+1)]
    if "DUP" in set(df_cnvs['CNV']):
        haplotypes.append(
                    ("h2",
                    df_cnvs[df_cnvs["CNV"] == "DUP"].copy(),
                    dir_fq / f"h2_{str(seed+2)}",
                    seed+2))
        if hom:
            haplotypes.append(
                    ("h3",
                    df_cnvs[df_cnvs["CNV"] == "DUP"].copy(),
                    dir_fq / f"h3_{str(seed+3)}",
                    seed+3))
    # generate fastq-files with multiprocessing
    #   writing to f"{path_fastq_stem}_1.fastq.gz", f"{path_fastq_stem}_2.fastq.gz"
    haplotype_seq_mp(
                        haplotypes=haplotypes,
                        path_wdir=pathlib.Path(workdir),
                        path_reference=pathlib.Path(fasta_reference),
                        path_probes=pathlib.Path(aligned_probes),
                        n_fragments=nfragments,
                        cores=cores)
    
    # define tuples of chr and df to correctly filter alignemnts
    # do correct merge
    dir_tmp_aln = tempfile.TemporaryDirectory()
    dir_aln = pathlib.Path(dir_tmp_aln.name)
    merge_files = []
    for h,df,path,seedi in haplotypes:
        fq_paths = list(map(pathlib.Path,[str(path)+'_1.fastq.gz',str(path)+'_2.fastq.gz']))
        alnpath = dir_aln / '.'.join([h,str(seedi),'unfiltered','bam'])
        align_pe_reads_to_ref(path_ref=fasta_reference,paths_reads=fq_paths, path_bam=alnpath,cores=cores)
        if df.empty: # in case no CNVs were selected, create only WT base files with full coverage
            merge_files.append(alnpath)
            continue
        outalnpath = dir_aln / '.'.join([h,str(seedi),'bam'])
        filter_bam(bam_file=alnpath, cnvs_df=df, outbam=outalnpath)
        merge_files.append(outalnpath)
    cmd_merge = shlex.split(f"samtools merge -f {str(bam_path)} {' '.join(list(map(str,merge_files)))}")
    subprocess.check_call(cmd_merge)
    cmd_index = shlex.split(f"samtools index {str(bam_path)}")
    subprocess.check_call(cmd_index)
    # document true CNVs -> "cnvs_{seed}.tsv"

#%%
# generate orders
def generate_call(wdir,panel,seed,cores=2,nfragments=0,dels=0,dups=0,hom=False,min_exons=0,max_exons=0):
    generate_true_cnv(
        workdir=wdir,
        panel=panel,
        seed=seed,
        dels=dels,
        dups=dups,
        with_freq=False,
        min_exons=min_exons,
        max_exons=max_exons,
        hom=hom)
    return f"python3 scripts/simulate_sample.py generate_sample \
        --workdir . \
        --panel {panel} \
        --seed {seed} \
        --bam_path samples/{panel}/{panel}.{seed}.bam \
        --fasta_reference REF/hs37d5.fa \
        --aligned_probes probes_aligned/{panel}.probes.bam \
        --arr_fragments coverage_distributions/arr.pckl \
        --cores {cores} \
        --nfragments {nfragments} \
        --dels {dels} \
        --dups {dups} \
        {'--hom' if hom else ''} \
        --min_exons {min_exons} \
        --max_exons {max_exons}", str(pathlib.Path(wdir) / f"samples/{panel}/{panel}.{seed}.bam")


def add_commands(wdir,commands_path,alias,start_seed,num_samples,panel,cores=2,nfragments=0,dels=0,dups=0,distr_gemoetric=False,hom=False,min_exons=0,max_exons=0,additional_samples=[]):
    l = []
    seed_counter = start_seed
    for i in range(num_samples):
        l.append(generate_call(wdir,panel,seed=seed_counter,cores=cores,
            nfragments=nfragments,
            dels=np.random.geometric(0.5,1)[0] if distr_gemoetric else dels,
            dups=np.random.geometric(0.5,1)[0] if distr_gemoetric else dups,
            hom=hom,
            min_exons=min_exons,
            max_exons=max_exons))
        seed_counter +=4
    # save to file
    path = pathlib.Path(commands_path).parent / panel / f"{alias}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w') as f:
        for line in [*additional_samples,*l]:
            print(line[1] ,file=f)
    return l,seed_counter


def generate_all_calls(wdir,commands_path,startseed,panel,cores=2,**kwargs):
    all_calls = []
    seed_num = startseed
    nums = [60,20,[3,5,8,12,17,23,30],range(20)] if panel != "BM" else [16,5,[5,17],range(3)]
    l,seed_num = add_commands(wdir,commands_path,"wild_types",start_seed=seed_num,num_samples=nums[0],panel=panel,cores=cores,
                                                                nfragments=0,dels=0,dups=0,hom=False,min_exons=0,max_exons=0)
    wild_type_samples = l
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"variants_het",start_seed=seed_num,num_samples=nums[0],panel=panel,cores=cores,
                                                                nfragments=0,dels=1,dups=1,distr_gemoetric=True,hom=False,min_exons=0,max_exons=0,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dels_hom",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=1,dups=0,hom=True,min_exons=0,max_exons=0,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dups_hom",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=0,dups=1,hom=True,min_exons=0,max_exons=0,additional_samples=wild_type_samples)
    all_calls.append(l)
    
    l,seed_num = add_commands(wdir,commands_path,"dels_het_small",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=2,dups=0,hom=False,min_exons=1,max_exons=1,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dups_het_small",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=0,dups=2,hom=False,min_exons=1,max_exons=1,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dels_het_medium",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=1,dups=0,hom=False,min_exons=5,max_exons=10,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dups_het_medium",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=0,dups=1,hom=False,min_exons=5,max_exons=10,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dels_het_big",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=1,dups=0,hom=False,min_exons=15,max_exons=0,additional_samples=wild_type_samples)
    all_calls.append(l)

    l,seed_num = add_commands(wdir,commands_path,"dups_het_big",start_seed=seed_num,num_samples=nums[1],panel=panel,cores=cores,
                                                                nfragments=0,dels=0,dups=1,hom=False,min_exons=15,max_exons=0,additional_samples=wild_type_samples)
    all_calls.append(l)

    alias = "noisy_samples"
    l = []
    for j in nums[2]:
        for i in range(3):
            l.append(generate_call(wdir=wdir,panel=panel,seed=seed_num,cores=cores,
                nfragments=0,
                dels=j,
                dups=j,
                hom=False,
                min_exons=0,
                max_exons=0))
            seed_num +=4
    all_calls.append(l)
    (pathlib.Path(commands_path).parent / panel).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(commands_path).parent / panel / f"{alias}.txt","w") as f:
        for line in [*wild_type_samples,*l]:
            print(line[1] ,file=f)
    
    alias = "low_coverage_samples"
    l = []
    for i in nums[3]:
        for j in range(3):
            l.append(generate_call(wdir=wdir,panel=panel,seed=seed_num,cores=cores,
                nfragments=10000*(2*i+1),
                dels=1,
                dups=1,
                hom=False,
                min_exons=0,
                max_exons=0))
            seed_num +=4
    all_calls.append(l)
    # flatten list
    (pathlib.Path(commands_path).parent / panel).mkdir(parents=True, exist_ok=True)
    with open(pathlib.Path(commands_path).parent / panel / f"{alias}.txt","w") as f:
        for line in [*wild_type_samples,*l]:
            print(line[1] ,file=f)
    # save file containing all commands
    with open(pathlib.Path(commands_path), "w") as f:
        for line in [item[0] for sublist in all_calls for item in sublist]:
            print(line,file=f)
    return [item for sublist in all_calls for item in sublist], seed_num

# %%

def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # -- prepare_probes --- #
    parser_prepare_probes = subparsers.add_parser("prepare_probes",description="intersect probes.bed with panel-bed to get a bed-file of only on-target probes")
    parser_prepare_probes.add_argument("--path_reference", type=os.path.abspath,required=True , help="input: path_reference.")
    parser_prepare_probes.add_argument("--path_panel_bed", type=os.path.abspath,required=True , help="input: path to bed-file of panel.")
    parser_prepare_probes.add_argument("--path_probes_bed", type=os.path.abspath,required=True , help="input: path to bed-file of probes.")
    parser_prepare_probes.add_argument("--path_outprobes", type=os.path.abspath,required=True , help="output: path to bam-file of selected probes.")
    parser_prepare_probes.add_argument("--cores", type=int,required=False,default=3 , help="param: number of threads used by minimap2.")
    parser_prepare_probes.set_defaults(func=prepare_probes)

    # -- generate_sample --- #
    parser_generate_sample = subparsers.add_parser("generate_sample",description="generates aligned simulated paired end reads.")
    parser_generate_sample.add_argument("--workdir", type=os.path.abspath,required=True , help="input: path to workdir.")
    parser_generate_sample.add_argument("--panel", type=str,required=True , help="param: panel name.")
    parser_generate_sample.add_argument("--seed", type=int,required=True , help="param: seed for this sample. should be divisible by four.")
    parser_generate_sample.add_argument("--bam_path", type=os.path.abspath,required=True , help="output: path to final aligned reads.")
    parser_generate_sample.add_argument("--fasta_reference", type=os.path.abspath,required=True , help="input: path to reference.")
    parser_generate_sample.add_argument("--aligned_probes", type=os.path.abspath,required=True , help="input: path to probes.bam.")
    parser_generate_sample.add_argument("--arr_fragments", type=os.path.abspath,required=False , help="input: pickled dictionary of read lengths per panel.")
    parser_generate_sample.add_argument("--nfragments", type=int,required=False,default=0 , help="param: number fragments. if another value than 0 is passed, this will overwrite 'arr_fragments'.")
    parser_generate_sample.add_argument("--dels", type=int,required=False,default=0 , help="param: number of deletions of the final sample.")
    parser_generate_sample.add_argument("--dups", type=int,required=False,default=0 , help="param: number of duplications of the final sample.")
    parser_generate_sample.add_argument("--with_freq",required=False,default=False, action="store_true" , help="param: link sampling probability of a CNV to it's relative frequency in gonmadSV data set.")
    parser_generate_sample.add_argument("--min_exons", type=int,required=False,default=0 , help="param: minimum number of exons per CNV.")
    parser_generate_sample.add_argument("--max_exons", type=int,required=False,default=0 , help="param: maximum number of exons per CNV.")
    parser_generate_sample.add_argument("--hom",required=False,default=False, action="store_true", help="param: if True, all CNVs are homozygous.")
    parser_generate_sample.add_argument("--cores", type=int,required=False,default=3 , help="param: number of threads used by minimap2.")
    parser_generate_sample.set_defaults(func=sample_gen)

    # -- generate_commands --- #
    parser_generate_commands = subparsers.add_parser("generate_commands",description="generates the commands to simulate all reads")
    parser_generate_commands.add_argument("--wdir", type=os.path.abspath,required=True , help="param: workdir.")
    parser_generate_commands.add_argument("--commands_path", type=os.path.abspath,required=True , help="output: to file which will contain all commands.")
    parser_generate_commands.add_argument("--panel", type=str,required=True , help="param: panel name.")
    parser_generate_commands.add_argument("--cores", type=int,required=False,default=2 , help="param: number of threads used in mp and by minimap2. Keep at 2 atm to avoid running low on memory.")
    parser_generate_commands.add_argument("--startseed", type=int,required=False,default=0 , help="param: starting seed. increments by 4 fro every sample.")
    parser_generate_commands.set_defaults(func=generate_all_calls)

    # -- generate_true_cnv --- #
    parser_generate_true_cnv = subparsers.add_parser("generate_true_cnv",description="generates the commands to simulate all reads")
    parser_generate_true_cnv.add_argument("--wdir", type=os.path.abspath,required=True , help="param: workdir.")
    parser_generate_true_cnv.add_argument("--panel", type=str,required=True , help="param: name of panel.")
    parser_generate_true_cnv.add_argument("--seed", type=int,required=True , help="param: seed for this sample. should be divisible by four.")
    parser_generate_true_cnv.add_argument("--dels", type=int,required=False,default=0 , help="param: number of deletions of the final sample.")
    parser_generate_true_cnv.add_argument("--dups", type=int,required=False,default=0 , help="param: number of duplications of the final sample.")
    parser_generate_true_cnv.add_argument("--with_freq",required=False,default=False, action="store_true" , help="param: link sampling probability of a CNV to it's relative frequency in gonmadSV data set.")
    parser_generate_true_cnv.add_argument("--min_exons", type=int,required=False,default=0 , help="param: minimum number of exons per CNV.")
    parser_generate_true_cnv.add_argument("--max_exons", type=int,required=False,default=0 , help="param: maximum number of exons per CNV.")
    parser_generate_true_cnv.add_argument("--hom",required=False,default=False, action="store_true", help="param: if True, all CNVs are homozygous.")
    parser_generate_true_cnv.set_defaults(func=generate_true_cnv)

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
    else:
        logger.info("Starting analysis...")
        args.func(**vars(args))
        logger.info("All done. Have a nice day!")

if __name__ == "__main__":
    main()

#%%

"""
panels = ["BM","CBM","CBM2","SDAG1","SDAG2","TAADv1","TAADv2"]
# %%
path_wdir = pathlib.Path("/fast/work/users/vmay_m/workflow/clearCNV_sim/artificial_genomes")
seed_num = 0
all_calls = {}
all_samples = []
for panel in panels:
    commands_path = path_wdir / f"commands/{panel}_commands.txt"
    commands, seed_num = generate_all_calls(
            wdir=path_wdir
            commands_path=commands_path,
            panel=panel,
            start_seed=seed_num,cores=2)
    all_calls[panel]={t[1]:t[0] for t in commands}
    all_samples = [*all_samples, *[t[1] for t in commands]]
"""
# %%
"""
import random
import matplotlib.pyplot as plt
workdir = pathlib.Path("/fast/work/users/vmay_m/workflow/clearCNV_sim/artificial_genomes")

def id_cnv(x):
    return '-'.join([str(x.chr),str(x.start),str(x.end),str(x.CNV)])

l = []
for i in range(100):
    x = get_cnvs_of_order(workdir, "SDAG1",seed=int(random.random()*2**16),cnvs=(1,1),with_freq=False,min_exons=5,max_exons=10,hom=False)
    l.append(id_cnv(x))

plt.hist([l.count(c) for c in set(l)])
"""

# %%
