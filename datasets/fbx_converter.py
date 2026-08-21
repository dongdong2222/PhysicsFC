
import fbx
import FbxCommon
from pathlib import Path


######################
# create directory
#####################
save_dir = Path('./soccer_mega_pack')
if not save_dir.exists():
    save_dir.mkdir()
    print('creted directory')
else:
    ...

data_path = Path('./AA_Soccer_Mega_Animations_Pack.fbx')


#######################
# binary import
######################

# Create an SDK manager
manager = fbx.FbxManager.Create()

# Create an IOSettings object.
ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
manager.SetIOSettings(ios)

# Create a new scene so it can be populated by the imported file.
scene = fbx.FbxScene.Create(manager,"myScene")

importer = fbx.FbxImporter.Create(manager, 'importer')

if not importer.Initialize(str(data_path), -1, manager.GetIOSettings()):
    print('failed to initialize')

if not importer.Import(scene):
    exit('failed to import')

print('imported original fbx file')

exporter = fbx.FbxExporter.Create(manager, '')
exporter.Initialize(str('AA_Soccer_Mega_Animations_Pack_ascii.fbx'), 4, manager.GetIOSettings()) # 1: ascii_2, 4: ascii_6
exporter.Export(scene)

print('exported ascii fbx file')


###########################
# modify
###########################
with open(str('AA_Soccer_Mega_Animations_Pack_ascii.fbx'), 'r') as f:
    ascii_data = f.read()
    prefix, *anims = ascii_data.split('Take: "')

    for anim in anims:
        title_end = anim.find(' ')
        title = anim[:title_end-1]
        merged_ascii = prefix + 'Take: "' + anim

        with open(str(save_dir / (title+'.fbx')), 'w') as f:
            f.write(merged_ascii)
            print(title, 'ascii file wrote')
        

#######################
# convert to binary
########################
for path in save_dir.glob('*'):

    manager = fbx.FbxManager.Create()

    # Create an IOSettings object.
    ios = fbx.FbxIOSettings.Create(manager, fbx.IOSROOT)
    manager.SetIOSettings(ios)

    # Create a new scene so it can be populated by the imported file.
    scene = fbx.FbxScene.Create(manager,"myScene")

    importer = fbx.FbxImporter.Create(manager, 'importer')

    if not importer.Initialize(str(path), -1, manager.GetIOSettings()):
        print('failed init')

    if not importer.Import(scene):
        print('failed import')

    exporter = fbx.FbxExporter.Create(manager, '')
    exporter.Initialize(str(path), -1, manager.GetIOSettings())
    exporter.Export(scene)

    print(path, 'binary file wrote')

print('completed')