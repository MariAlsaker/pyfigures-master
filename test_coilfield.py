import coil_field

single_loop_coil = coil_field.Coil_field(15, 100)
#single_loop_coil.show_field("B")
single_loop_coil.show_field_slice("B", 50)
single_loop_coil.show_field_slice("B", 97)