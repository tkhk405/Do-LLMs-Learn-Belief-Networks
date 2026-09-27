"""Shared comparison-figure layout preserved from the submission renderer.

Callers provide matrices in the canonical six-issue order. No input files are
read and no figures are written by this module.
"""
import numpy as np
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from itertools import permutations
labels=['Defense','Social welfare','Public works','Fiscal stimulus','North Korea','Public safety']
codes=['De','SW','PW','FS','NK','PS']
ix=np.triu_indices(6,1)
pairs=[codes[i]+'-'+codes[j] for i,j in zip(*ix)]
statistics=[]
def matrix(fig,rect,m,letter,compact=False):
 ax=fig.add_axes(rect);im=ax.imshow(m,cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');ax.set_xticks(range(6),[z.replace(' ','\n',1) for z in labels] if compact else labels,rotation=90);ax.set_yticks(range(6),[z.replace(' ','\n',1) for z in labels] if compact else labels);ax.tick_params(length=0,pad=3)
 for i in range(6):
  for j in range(6):ax.text(j,i,f'{m[i,j]:.2f}'.replace('-','−'),ha='center',va='center',fontsize=8,color='white' if abs(m[i,j])>.55 else 'black')
 ax.set_title(f'({letter})',loc='left',fontweight='bold',fontsize=11)
 fig.canvas.draw();pos=ax.get_position()
 if compact:cb=fig.colorbar(im,cax=fig.add_axes([pos.x1+.006,pos.y0,.009,pos.height]),ticks=[-1,0,1])
 else:cb=fig.colorbar(im,cax=fig.add_axes([pos.x1+.012,pos.y0,.014,pos.height]),ticks=[-1,-.5,0,.5,1])
 cb.ax.tick_params(labelsize=8)
 return ax

def scatter(fig,rect,a,b,xlabel,ylabel,letter,expected,compact=False):
 ax=fig.add_axes(rect);x=a[ix];y=b[ix];rho=spearmanr(x,y).statistic;p=np.mean([spearmanr(x,b[np.ix_(q,q)][ix]).statistic>=rho-1e-12 for q in permutations(range(6))]);assert (round(rho,3),round(p,3))==expected
 statistics.append({'panel':letter,'x':xlabel,'y':ylabel,'rho':rho,'p':p})
 ax.scatter(x,y,s=25,c='#2980b9',edgecolors='white',linewidths=.5,zorder=5);line=np.linspace(-.4,.95,100);ax.plot(line,np.polyval(np.polyfit(x,y,1),line),'--',color='#c0392b',lw=1,alpha=.7)
 ax.set(xlim=(-.45,1),ylim=(-.45,1),xlabel=xlabel,ylabel=ylabel);ax.set_aspect('equal');ax.set_xticks([-.25,0,.25,.5,.75,1]);ax.set_yticks([-.25,0,.25,.5,.75,1]);ax.grid(alpha=.2);ax.axhline(0,color='grey',lw=.4,ls=':');ax.axvline(0,color='grey',lw=.4,ls=':');ax.set_title(f'({letter})',loc='left',fontweight='bold',fontsize=11)
 # Place callouts on a grid with hard exclusion around every data marker.
 fig.canvas.draw();renderer=fig.canvas.get_renderer()
 stat_text=(rf'$\rho = {rho:.3f}$'+'\n'+rf'$p = {p:.3f}$') if compact else rf'$\rho = {rho:.3f},\ p = {p:.3f}$'
 stat=ax.text(.97,.025,stat_text,transform=ax.transAxes,ha='right',va='bottom',fontsize=8,bbox=dict(fc='white',ec='.6',lw=.4,pad=2),zorder=7)
 fig.canvas.draw();sb=stat.get_bbox_patch().get_window_extent(renderer).expanded(1.04,1.08)
 points=np.column_stack([x,y]);pd=ax.transData.transform(points)
 probe=ax.text(0,0,'SW-PW',fontsize=8);fig.canvas.draw();bb=probe.get_window_extent(renderer);hw=bb.width/2+3;hh=bb.height/2+2;probe.remove()
 from matplotlib.transforms import Bbox
 best=[]
 for dx in [-.025,0,.025]:
  for dy in [-.035,0,.025]:
   grid=np.array([(xx+dx,yy+dy) for yy in np.linspace(-.26,.90,9) for xx in [-.24,.26,.76]])
   keep=[]
   for k,(cx,cy) in enumerate(ax.transData.transform(grid)):
    box=Bbox.from_extents(cx-hw,cy-hh,cx+hw,cy+hh)
    if box.overlaps(sb):continue
    if any(box.expanded(1+6/(2*hw),1+6/(2*hh)).contains(*pt) for pt in pd):continue
    keep.append(k)
   if len(keep)>len(best):best=grid[keep]
 candidates=np.array(best);assert len(candidates)>=15,('Not enough free callout positions',letter,len(candidates))
 cost=((points[:,None,:]-candidates[None,:,:])**2).sum(2);rows,cols=linear_sum_assignment(cost);texts=[]
 for i,j in zip(rows,cols):texts.append(ax.annotate(pairs[i],(x[i],y[i]),xytext=candidates[j],ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.1',fc='white',ec='.6',lw=.4),arrowprops=dict(arrowstyle='->',color='.6',lw=.5,shrinkB=4),zorder=6))
 fig.canvas.draw();boxes=[z.get_bbox_patch().get_window_extent(renderer) for z in texts+[stat]]
 assert not [(i,j) for i in range(len(boxes)) for j in range(i+1,len(boxes)) if boxes[i].overlaps(boxes[j])]
 assert all(not b.expanded(1.04,1.1).contains(*pt) for b in boxes for pt in pd)

 return ax
