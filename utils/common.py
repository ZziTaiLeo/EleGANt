from PIL import Image
import matplotlib.pyplot as plt


# Log images
def log_input_image(x, opts):
	return tensor2im(x)


def tensor2im(var):
	# var shape: (3, H, W)
	var = var.cpu().detach().transpose(0, 2).transpose(0, 1).numpy()
	var = ((var + 1) / 2)
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def vis_faces(log_hooks):
	display_count = len(log_hooks)
	fig = plt.figure(figsize=(8, 4 * display_count))
	gs = fig.add_gridspec(display_count, len(log_hooks[0]))
	print('log_hooks:', log_hooks)
	for i in range(display_count):
		hooks_dict = log_hooks[i]
		fig.add_subplot(gs[i, 0])
		if 'diff_input' in hooks_dict:
			vis_faces_with_id(hooks_dict, fig, gs, i)
		else:
			vis_faces_no_id(hooks_dict, fig, gs, i)
	plt.tight_layout()
	return fig


def vis_faces_with_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['source_face'])
    plt.title('Input\nOut Sim={:.2f}'.format(float(hooks_dict['diff_input'])))
    col = 0
    if 'reference_face' in hooks_dict:
        col += 1
        fig.add_subplot(gs[i, col])
        plt.imshow(hooks_dict['reference_face'])
        plt.title('Reference\nIn={:.2f}, Out={:.2f}'.format(float(hooks_dict['diff_views']),
	                                                 float(hooks_dict['diff_target'])))
    col += 1
    fig.add_subplot(gs[i, col])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))

    if 'pgt_face' in hooks_dict:
        col+=1
        fig.add_subplot(gs[i, col])
        plt.imshow(hooks_dict['pgt_face'])
        plt.title('PgtFace\n Target Sim={:.2f}'.format(float(hooks_dict['diff_target'])))



def vis_faces_no_id(hooks_dict, fig, gs, i):
    plt.imshow(hooks_dict['source_face'])
    plt.title('Input')
    col = 0
    if 'reference_face' in hooks_dict:
        col += 1
        fig.add_subplot(gs[i, col])
        plt.imshow(hooks_dict['reference_face'])
        plt.title('Reference')

    col += 1
    fig.add_subplot(gs[i, col])
    plt.imshow(hooks_dict['output_face'])
    plt.title('Output')

    if 'pgt_face' in hooks_dict:
        col+=1
        fig.add_subplot(gs[i, col])
        plt.imshow(hooks_dict['pgt_face'])
        plt.title('pgt_face')

