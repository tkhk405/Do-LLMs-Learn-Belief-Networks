"""Compare internal representation matrices with each other and local UTAS data.
The figures operation uses saved matrices and does not load respondent records.
"""
from pathlib import Path
import argparse,json,hashlib,importlib.metadata
import numpy as np,pandas as pd
from common.configuration import load_paths
from common.constants import ISSUES,ISSUE_COLUMNS_JP
from common.data_loading import load_matrix,clean_utas_frame
from common.statistics import utas_matrix,exact_mantel

def compare(args):
    paths=load_paths(args.config)
    if args.output.exists():raise FileExistsError('Choose a fresh output directory')
    transfer_path=args.transfer;cosine_path=args.cosine
    raw=load_matrix(transfer_path);transfer=(raw+raw.T)/2;cosine=load_matrix(cosine_path)
    utas={};counts={};inputs=[transfer_path,cosine_path,Path(__file__)]
    for sample in ['elected','all_candidates']:
        file=paths['utas_'+sample+'_csv'];frame=clean_utas_frame(file)
        utas[sample]=utas_matrix(frame)
        valid=frame[ISSUE_COLUMNS_JP].notna().astype(int);counts[sample]=(len(frame),(valid.T@valid).to_numpy());inputs.append(file)
    comparisons=[('transfer_vs_cosine',transfer,cosine,None)]
    for sample in utas:
        comparisons.extend([(f'transfer_vs_utas_{sample}',transfer,utas[sample],sample),(f'cosine_vs_utas_{sample}',cosine,utas[sample],sample)])
    rows=[];nulls={}
    for name,a,b,sample in comparisons:
        result,null=exact_mantel(a,b);rows.append({'model':args.model,'comparison':name,**result,'utas_records':counts[sample][0] if sample else None});nulls[name]=null
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/'matrix_comparisons.csv',index=False)
    np.savez(args.output/'permutation_distributions.npz',**nulls)
    for sample,m in utas.items():
        pd.DataFrame(m,index=ISSUES,columns=ISSUES).to_csv(args.output/f'utas_{sample}_spearman.csv')
        pd.DataFrame(counts[sample][1],index=ISSUES,columns=ISSUES).to_csv(args.output/f'utas_{sample}_pairwise_n.csv')
    manifest={'versions':{k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy']},'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'test':'one-sided greater-or-equal, exhaustive joint row/column permutations including identity','transfer_symmetrization':'(T + T.T) / 2','p_values':'unrounded'}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

def figures(a):
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 from PIL import Image
 from common import plotting as layout
 if a.output.exists():raise FileExistsError('Choose a fresh output directory')
 config=json.loads(a.inputs.read_text());data={};sources={}
 for model in ['gemma','llama']:
  data[model]={}
  for field in ['transfer','cosine','elected','all_candidates']:
   path=Path(config[model][field]).expanduser()
   if not path.is_absolute():path=a.inputs.resolve().parent/path
   data[model][field]=load_matrix(path)
   sources[model+'.'+field]=hashlib.sha256(path.read_bytes()).hexdigest()
  symmetric=(data[model]['transfer']+data[model]['transfer'].T)/2
  if 'symmetric' in config[model]:
   path=Path(config[model]['symmetric']).expanduser()
   if not path.is_absolute():path=a.inputs.resolve().parent/path
   saved=load_matrix(path)
   if not np.allclose(saved,symmetric,atol=1e-12,rtol=0):raise ValueError('Saved symmetric transfer disagrees with directional input')
   symmetric=saved
   sources[model+'.symmetric']=hashlib.sha256(path.read_bytes()).hexdigest()
  data[model]['symmetric']=symmetric
 plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'mathtext.fontset':'custom','mathtext.rm':'Arial','mathtext.it':'Arial:italic','mathtext.bf':'Arial:bold','axes.unicode_minus':True})
 a.output.mkdir(parents=True)
 records={}
 def save(fig,name):
  fig.savefig(a.output/(name+'.pdf'));fig.savefig(a.output/(name+'.png'),dpi=600)
  with Image.open(a.output/(name+'.png')) as image:image.convert('RGB').save(a.output/(name+'.tif'),compression='tiff_lzw',dpi=(600,600))
  records[name]=list(layout.statistics);layout.statistics.clear();plt.close(fig)
 for model,names,expected in [('gemma',['Fig1','Fig4','Fig5','Fig6'],[(.529,.015),(.975,.003),(.868,.003),(.850,.006),(.611,.022),(.625,.019)]),('llama',['S1_Fig','S6_Fig','S7_Fig','S10_Fig'],[(.364,.078),(.939,.001),(.775,.003),(.671,.006),(.493,.078),(.418,.132)])]:
  d=data[model];t,c,u,all_u,ts=[d[k] for k in ['transfer','cosine','elected','all_candidates','symmetric']]
  for name,m1,m2,x,y,xlab,ylab,stat in [(names[1],t,c,c,ts,'Cosine similarity','Transfer performance',expected[1])]:
   fig=plt.figure(figsize=(7.5,3.3));layout.matrix(fig,[.085,.29,.23,.55],m1,'A',True);layout.matrix(fig,[.415,.29,.23,.55],m2,'B',True);layout.scatter(fig,[.765,.29,.22,.55],x,y,xlab,ylab,'C',stat,True);save(fig,name)
  fig=plt.figure(figsize=(7.5,4));layout.scatter(fig,[.09,.18,.37,.70],ts,u,'Transfer performance','UTAS correlation','A',expected[2]);layout.scatter(fig,[.60,.18,.37,.70],c,u,'Cosine similarity','UTAS correlation','B',expected[3]);save(fig,names[2])
  fig=plt.figure(figsize=(7.5,3.3));layout.matrix(fig,[.085,.29,.23,.55],all_u,'A',True);layout.scatter(fig,[.445,.29,.22,.55],ts,all_u,'Transfer performance','UTAS correlation','B',expected[4],True);layout.scatter(fig,[.765,.29,.22,.55],c,all_u,'Cosine similarity','UTAS correlation','C',expected[5],True);save(fig,names[3])
 (a.output/'verification.json').write_text(json.dumps({'source_sha256':sources,'statistics':records,'annotation_overlap_assertions':'passed','expected_rounded_statistics':'passed'},indent=2))
 print('Rendered six internal-representation comparison figures')

def main():
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='operation',required=True)
 q=sub.add_parser('compare');q.add_argument('--config');q.add_argument('--model',required=True,choices=['gemma','llama']);q.add_argument('--transfer',type=Path,required=True);q.add_argument('--cosine',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 q=sub.add_parser('figures');q.add_argument('--inputs',type=Path,required=True);q.add_argument('--output',type=Path,required=True)
 a=p.parse_args()
 if a.operation=='compare':compare(a)
 else:figures(a)
if __name__=='__main__':main()
