import coil_field

my_coil = coil_field.Coil_field(15, 100)
#my_coil.save_visualized_field("B")
my_coil.show_field("B", gif_name="My_coil_field.mp4")
#my_coil.show_field_slice("B", 50)