def main() -> None:
    from buscador_textual import Buscador

    app = Buscador("vocabulario_index.json", "parrafos_index.json")
    app.run()


if __name__ == "__main__":
    main()
