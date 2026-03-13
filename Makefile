PROTO_DIR=/usr/share/wayland-protocols

main: main.c \
				xdg-shell-protocol.c \
				xdg-shell-protocol.h \
				shaders/main_vert.spv.h \
				shaders/main_frag.spv.h
	gcc -g -Wall -lwayland-client -lwayland-cursor -lvulkan -lm -o $@ \
		main.c \
		xdg-shell-protocol.c

shaders/main_vert.spv.h: shaders/main.vert
	glslangValidator -o $@ --variable-name main_vert -V $<

shaders/main_frag.spv.h: shaders/main.frag
	glslangValidator -o $@ --variable-name main_frag -V $<

xdg-shell-protocol.c:
	wayland-scanner private-code $(PROTO_DIR)/stable/xdg-shell/xdg-shell.xml $@

xdg-shell-protocol.h:
	wayland-scanner client-header $(PROTO_DIR)/stable/xdg-shell/xdg-shell.xml $@
