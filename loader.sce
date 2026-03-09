
getd("macros")

abort

mode(-1);
//getd(".")

function p=full_path(p)
  if MSDOS then
    if part(p,2)<>':' then p=pwd()+'\'+p,end
  else
    if part(p,1)<>'/' then p=pwd()+'/'+p,end
  end
endfunction

Mpath=get_file_path('loader.sce');
if Mpath=='.' then Mpath='',end
Mpath=full_path(Mpath);

genlib('FemLablib',Mpath+'macros')

v = getversion("scilab");

if v(1)<5
  mkhelpflag=%t;

  // check if help files are already loaded
  for k=size(%helps,1):-1:1
    if (%helps(k,2)=='Femlab: Finite Element Laboratory')
      mkhelpflag=%f;
      break;
    end
  end
  if mkhelpflag
    %helps=[%helps; [ Mpath+'help' 'Femlab: Finite Element Laboratory']];
  end

disp("Hit ENTER to continue ...")

end
clear Mpath full_path mkhelpflag v
