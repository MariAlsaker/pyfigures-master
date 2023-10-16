import coil_field

# single_loop_coil = coil_field.Coil_field(15, 100)
# single_loop_coil.show_field_lines(slice="y99")
# single_loop_coil.show_field("B", gif_name="norm_to_mid_val.mp4")
# single_loop_coil.show_field_slice("B", slice=50, verbose=True, filename="center_norm")
# single_loop_coil.show_field_slice("B", slice=97, verbose=True, filename="plus75mm_norm")
# single_loop_coil.show_field_slice("B", slice=2, verbose=True, filename="minus75mm_norm")

quad_coil = coil_field.Coil_field(15, 100, quadrature=True)
quad_coil.show_coil()
quad_coil.show_field_lines(slice="z50")
#quad_coil.show_field_magnitude("B", gif_name="quad_norm_to_mid_val.mp4")