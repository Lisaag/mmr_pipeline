import glob, os
import vedo
from vedo import shapes, plotter
import settings

def list_db_files(dir) -> list[str]:
    files = glob.glob(f'{dir}/**/*.obj')
    files.sort()
    return files

def create_output_folder(folder_name: str) -> None:
    if os.path.isdir(folder_name):
        return
    
    os.mkdir(folder_name)

def show_mesh_plot(mesh: vedo.Mesh, show_unit_cube: bool, mesh_color: str = '#de9273') -> None:
    def check_quit(event: plotter.Event):
        if (event.keyPressed == 'q'):
            exit(0)
    mesh_color = 'white'

    mesh.linecolor(lc='orange')
    mesh.linewidth(lw=1.0)

    mesh.wireframe(value=True)
    plt = vedo.Plotter()
    plt.add_callback('KeyPress', check_quit)

    
    # const axes object for every new plot
    AXES = vedo.Axes(
        (0,0,0),
        xtitle='x',
        ytitle='y',
        ztitle='z',
        xrange=[0,1],
        yrange=[0,1],
        zrange=[0,1],
    )

    # Show mesh in plot
    mesh.color(mesh_color) # type: ignore
    plt.add(mesh)
    
    if show_unit_cube:
        cube = shapes.Cube(mesh.GetOrigin(), side=1, c=('blue'), alpha=0.1)
        cube.wireframe(value=True)
        plt.add(cube)

    #plt.add(AXES)
    plt.show()    
