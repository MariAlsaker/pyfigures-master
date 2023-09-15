import coil_field

single_loop_coil = coil_field.Coil_field(15, 100)
#my_coil.save_visualized_field("B")
single_loop_coil.show_field("B", vmax=100)
#my_coil.show_field_slice("B", 50)