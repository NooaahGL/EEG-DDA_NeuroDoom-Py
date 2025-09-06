class UserProfile:
    """
    Perfil de usuario que almacena metadata fija y estadísticas
    en tiempo real de las muestras recibidas.
    """
    def __init__(self, name: str, age: int, gender: str):
        # metadata
        self.name = name
        self.age = age
        self.gender = gender

        # estadísticas iniciales
        self.count = 0
        self.att_sum = 0.0
        self.med_sum = 0.0
        self.sig_sum = 0.0

        self.att_min = None
        self.att_max = None
        self.med_min = None
        self.med_max = None
        self.sig_min = None
        self.sig_max = None

        

    def add_sample(self, attention: float, meditation: float, signal: float):
        """Incorpora una nueva muestra y actualiza estadísticas."""
        self.count += 1

        # sumas
        self.att_sum += attention
        self.med_sum += meditation
        self.sig_sum += signal

        # mínimos/máximos
        def _upd(val, current_min, current_max):
            if current_min is None or val < current_min:
                current_min = val
            if current_max is None or val > current_max:
                current_max = val
            return current_min, current_max

        self.att_min, self.att_max = _upd(attention, self.att_min, self.att_max)
        self.med_min, self.med_max = _upd(meditation, self.med_min, self.med_max)
        self.sig_min, self.sig_max = _upd(signal, self.sig_min, self.sig_max)

    @property
    def att_mean(self) -> float:
        return self.att_sum / self.count if self.count else 0.0

    @property
    def med_mean(self) -> float:
        return self.med_sum / self.count if self.count else 0.0

    def summary(self) -> dict:
        """Devuelve un diccionario con las estadísticas actuales."""
        return {
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "samples": self.count,
            "attention": {
                "mean": self.att_mean,
                "min": self.att_min,
                "max": self.att_max,
            },
            "meditation": {
                "mean": self.med_mean,
                "min": self.med_min,
                "max": self.med_max,
            },
            "signal": {
                "min": self.sig_min,
                "max": self.sig_max,
            },
        }
