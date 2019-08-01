import jupytext
import glob, os


currDir = os.getcwd()
print(currDir)
from_dir = os.path.join(currDir,  "ML", "USL")
#from_dir = os.getcwd()
#os.path.normpath('./')
print("PAth is: " + from_dir)
to_dir = os.path.join(from_dir, 'NB_Created')

def convert(pyfile):
  """
  Convert a preprocessed string object into notebook file
  """
  base = os.path.basename(pyfile)
  root, ext = os.path.splitext(base)
  target = os.path.join(to_dir, root + '.ipynb')
  print('Converting: ' + pyfile + ' >>> ' + target)
  nb = jupytext.readf(pyfile)
  jupytext.writef(nb, target, fmt='notebook')


if not os.path.exists(to_dir):
  """
  Create exprot directory, if it not exists
  """
  print('Directory ' + to_dir + ' created.')
  os.makedirs(to_dir)

for pyfile in glob.glob(os.path.join(from_dir , '*.py')):
  print(pyfile)
  if pyfile.endswith("LDA_Approach_1.py"):
    print('Here')
    convert(pyfile)
