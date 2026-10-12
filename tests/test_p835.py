"""
tests/itu/test_p835.py — Тесты валидации для ITU-R P.835-7.

Проверяем:
    1. Функции Annex 1 в ключевых точках (сверка с U.S. Standard 1976).
    2. Непрерывность T(Z) и P(Z) на границах регимов.
    3. Монотонность P(Z) — давление всегда падает.
    4. Round-trip: H(Z(H)) == H.
    5. Профили Annex 2 в ключевых точках.
    6. Сравнение с CSV-эталоном (если он есть).
"""

import math
import pytest

from p835 import (
    global_temperature,
    global_pressure,
    global_water_vapour_density,
    global_wet_pressure,
    convert_to_geopotential_height,
    convert_to_geometric_height,
    water_vapour_density_to_pressure,
    # Annex 2
    low_latitude_annual_temperature,
    low_latitude_annual_pressure,
    low_latitude_annual_water_vapour,
    mid_latitude_summer_temperature,
    mid_latitude_winter_temperature,
    high_latitude_summer_temperature,
    high_latitude_winter_temperature,
    interpolated_temperature,
    Season,
)


# =====================================================================
# 1. ЗНАЧЕНИЯ В КЛЮЧЕВЫХ ТОЧКАХ (Annex 1)
# =====================================================================


class TestAnnex1Values:
    """Проверка ключевых значений U.S. Standard Atmosphere 1976.

    ВАЖНО: границы слоёв P.835-7 заданы в геопотенциальной высоте H'.
    Публичные global_temperature()/global_pressure() принимают
    геометрическую высоту Z и конвертируют её сами. Поэтому в тестах
    на границах слоёв мы передаём Z = convert_to_geometric_height(H'_boundary)
    и ожидаем табличное значение для этой границы.
    """

    def test_temperature_at_sea_level(self):
        # T(0) = 288.15 K (по определению; Z=0 → H'=0)
        assert global_temperature(0.0) == pytest.approx(288.15, abs=1e-9)

    def test_temperature_at_tropopause(self):
        # H' = 11 км → T = 216.65 K
        z = convert_to_geometric_height(11.0)
        assert global_temperature(z) == pytest.approx(216.65, abs=1e-6)

    def test_temperature_at_20km(self):
        # H' = 20 км → T = 216.65 K
        z = convert_to_geometric_height(20.0)
        assert global_temperature(z) == pytest.approx(216.65, abs=1e-6)

    def test_temperature_at_32km(self):
        # H' = 32 км → T = 228.65 K
        z = convert_to_geometric_height(32.0)
        assert global_temperature(z) == pytest.approx(228.65, abs=1e-6)

    def test_temperature_at_47km(self):
        # H' = 47 км → T = 270.65 K
        z = convert_to_geometric_height(47.0)
        assert global_temperature(z) == pytest.approx(270.65, abs=1e-6)

    def test_temperature_at_51km(self):
        # H' = 51 км → T = 270.65 K
        z = convert_to_geometric_height(51.0)
        assert global_temperature(z) == pytest.approx(270.65, abs=1e-6)

    def test_temperature_at_71km(self):
        # H' = 71 км → T = 214.65 K
        z = convert_to_geometric_height(71.0)
        assert global_temperature(z) == pytest.approx(214.65, abs=1e-6)

    def test_pressure_at_sea_level(self):
        # P(0) = 1013.25 гПа
        assert global_pressure(0.0) == pytest.approx(1013.25, abs=1e-6)

    def test_pressure_at_20km(self):
        # H' = 20 км → P = 54.7498 гПа (граница слоя 11–20 км)
        z = convert_to_geometric_height(20.0)
        assert global_pressure(z) == pytest.approx(54.7498, abs=1e-2)

    def test_pressure_at_47km(self):
        # H' = 47 км → P = 1.109106 гПа
        z = convert_to_geometric_height(47.0)
        assert global_pressure(z) == pytest.approx(1.109106, abs=1e-3)

    def test_pressure_at_51km(self):
        # H' = 51 км → P = 0.6694167 гПа
        z = convert_to_geometric_height(51.0)
        assert global_pressure(z) == pytest.approx(0.6694167, abs=1e-4)

    def test_pressure_at_71km(self):
        # H' = 71 км → P = 0.03956649 гПа
        z = convert_to_geometric_height(71.0)
        assert global_pressure(z) == pytest.approx(0.03956649, abs=1e-5)


# =====================================================================
# 2. НЕПРЕРЫВНОСТЬ НА ГРАНИЦАХ
# =====================================================================

class TestContinuity:
    """Проверка непрерывности на границах слоёв."""

    BOUNDARIES_KM = [11.0, 20.0, 32.0, 47.0, 51.0, 71.0]

    @pytest.mark.parametrize("z_km", BOUNDARIES_KM)
    def test_temperature_continuous(self, z_km):
        eps = 1e-4
        T_left = global_temperature(z_km - eps)
        T_right = global_temperature(z_km + eps)
        # На границе T непрерывна
        assert abs(T_left - T_right) < 0.01  # K

    @pytest.mark.parametrize("z_km", BOUNDARIES_KM)
    def test_pressure_continuous(self, z_km):
        eps = 1e-4
        P_left = global_pressure(z_km - eps)
        P_right = global_pressure(z_km + eps)
        # Относительная невязка < 1e-3
        rel = abs(P_left - P_right) / max(P_left, 1e-12)
        assert rel < 1e-3

    def test_temperature_continuous_at_86km(self):
        """Граница между regime 1 и regime 2 (Z = 86 км)."""
        eps = 1e-3
        T_left = global_temperature(86.0 - eps)
        T_right = global_temperature(86.0 + eps)
        # В P.835-7 нет строгой непрерывности T на 86 км,
        # но разница не должна быть катастрофической
        assert abs(T_left - T_right) < 5.0  # K

    def test_pressure_continuous_at_86km(self):
        """Граница между regime 1 и regime 2 (Z = 86 км)."""
        eps = 1e-3
        P_left = global_pressure(86.0 - eps)
        P_right = global_pressure(86.0 + eps)
        rel = abs(P_left - P_right) / max(P_left, 1e-12)
        assert rel < 0.05  # 5% расхождение допустимо


# =====================================================================
# 3. МОНОТОННОСТЬ ДАВЛЕНИЯ
# =====================================================================

class TestMonotonicity:
    """P(Z) должна монотонно убывать."""

    def test_pressure_monotonically_decreasing(self):
        prev_P = global_pressure(0.0)
        for z_km in range(1, 100):
            P = global_pressure(float(z_km))
            assert P < prev_P, f"P({z_km}) = {P} >= P({z_km-1}) = {prev_P}"
            prev_P = P

    def test_pressure_positive(self):
        for z_km in range(0, 101):
            P = global_pressure(float(z_km))
            assert P > 0.0, f"P({z_km}) = {P} <= 0"


# =====================================================================
# 4. ROUND-TRIP ПРЕОБРАЗОВАНИЙ ВЫСОТ
# =====================================================================

class TestHeightConversions:
    """H(Z(H)) == H и Z(H(Z)) == Z."""

    @pytest.mark.parametrize("h_km", [0.0, 1.0, 10.0, 50.0, 80.0, 84.0, 84.852])
    def test_round_trip_h_to_z_to_h(self, h_km):
        z = convert_to_geometric_height(h_km)
        h2 = convert_to_geopotential_height(z)
        assert h2 == pytest.approx(h_km, abs=1e-9)

    @pytest.mark.parametrize("z_km", [0.0, 1.0, 10.0, 50.0, 80.0, 85.0, 86.0])
    def test_round_trip_z_to_h_to_z(self, z_km):
        h = convert_to_geopotential_height(z_km)
        z2 = convert_to_geometric_height(h)
        assert z2 == pytest.approx(z_km, abs=1e-9)

    def test_known_value_84_852(self):
        """H = 84.852 → Z = 86.000 (P.835-7)."""
        z = convert_to_geometric_height(84.852)
        assert z == pytest.approx(86.0, abs=0.01)

    def test_known_value_86(self):
        """Z = 86 → H ≈ 84.852."""
        h = convert_to_geopotential_height(86.0)
        assert h == pytest.approx(84.852, abs=0.01)


# =====================================================================
# 5. ВОДЯНОЙ ПАР
# =====================================================================

class TestWaterVapour:
    """Проверки профиля H₂O."""

    def test_water_vapour_at_sea_level_default(self):
        # ρ(0) = 7.5 г/м³ (по умолчанию)
        assert global_water_vapour_density(0.0) == pytest.approx(7.5, abs=1e-6)

    def test_water_vapour_decreases_with_height(self):
        prev = global_water_vapour_density(0.0)
        for z_km in range(1, 100):
            rho = global_water_vapour_density(float(z_km))
            assert rho <= prev + 1e-9, f"ρ({z_km}) = {rho} > ρ({z_km-1}) = {prev}"
            prev = rho

    def test_water_vapour_positive(self):
        for z_km in range(0, 101):
            rho = global_water_vapour_density(float(z_km))
            assert rho >= 0.0

    def test_pressure_from_density(self):
        # e = ρ·T / 216.7
        rho = 5.0
        T = 280.0
        e = water_vapour_density_to_pressure(rho, T)
        expected = 5.0 * 280.0 / 216.7
        assert e == pytest.approx(expected, abs=1e-9)

    def test_wet_pressure_positive(self):
        for z_km in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
            e = global_wet_pressure(z_km)
            assert e >= 0.0


# =====================================================================
# 6. ANNEX 2 — СЕЗОННЫЕ АТМОСФЕРЫ
# =====================================================================

class TestAnnex2:
    """Проверки сезонных профилей."""

    def test_low_annual_sea_level_temperature(self):
        # T(0) = 300.4222 K
        assert low_latitude_annual_temperature(0.0) == pytest.approx(300.4222, abs=1e-4)

    def test_low_annual_sea_level_pressure(self):
        # P(0) = 1012.0306 гПа
        assert low_latitude_annual_pressure(0.0) == pytest.approx(1012.0306, abs=1e-3)

    def test_low_annual_sea_level_water_vapour(self):
        # ρ(0) = 19.6542 г/м³
        assert low_latitude_annual_water_vapour(0.0) == pytest.approx(19.6542, abs=1e-3)

    def test_mid_summer_sea_level_temperature(self):
        assert mid_latitude_summer_temperature(0.0) == pytest.approx(294.9838, abs=1e-3)

    def test_mid_winter_sea_level_temperature(self):
        assert mid_latitude_winter_temperature(0.0) == pytest.approx(272.7241, abs=1e-3)

    def test_high_summer_sea_level_temperature(self):
        assert high_latitude_summer_temperature(0.0) == pytest.approx(286.8374, abs=1e-3)

    def test_high_winter_sea_level_temperature(self):
        assert high_latitude_winter_temperature(0.0) == pytest.approx(257.4345, abs=1e-3)

    def test_interpolation_at_low_latitude(self):
        """На 10°N интерполяция должна дать low annual."""
        z = 5.0
        T_low = low_latitude_annual_temperature(z)
        T_interp = interpolated_temperature(z, 10.0, Season.SUMMER)
        assert T_interp == pytest.approx(T_low, abs=1e-6)

    def test_interpolation_at_mid_latitude(self):
        """На 45°N интерполяция должна дать mid summer."""
        z = 5.0
        T_mid = mid_latitude_summer_temperature(z)
        T_interp = interpolated_temperature(z, 45.0, Season.SUMMER)
        assert T_interp == pytest.approx(T_mid, abs=1e-6)

    def test_interpolation_between_low_and_mid(self):
        """На 30°N — линейная интерполяция между low и mid."""
        z = 5.0
        T_low = low_latitude_annual_temperature(z)
        T_mid = mid_latitude_summer_temperature(z)
        T_interp = interpolated_temperature(z, 30.0, Season.SUMMER)
        expected = 0.5 * (T_low + T_mid)
        assert T_interp == pytest.approx(expected, abs=1e-6)


# =====================================================================
# 7. ГРАНИЦЫ ДИАПАЗОНА
# =====================================================================

class TestBoundaries:
    """Проверки граничных значений."""

    def test_negative_height_raises(self):
        from p835.common.exceptions import ITUValidationError
        with pytest.raises(ITUValidationError):
            global_temperature(-0.1)

    def test_height_above_100_raises(self):
        from p835.common.exceptions import ITUValidationError
        with pytest.raises(ITUValidationError):
            global_temperature(100.1)

    def test_z_0_and_100_ok(self):
        # На границах не должно быть исключений
        assert global_temperature(0.0) > 0
        assert global_temperature(100.0) > 0
        assert global_pressure(0.0) > 0
        assert global_pressure(100.0) > 0


# =====================================================================
# 8. СРАВНЕНИЕ С CSV-ЭТАЛОНОМ (если есть)
# =====================================================================

class TestReferenceCSV:
    """Сравнение с p835_reference.csv (если файл есть)."""

    def test_reference_csv(self, load_reference):
        rows = load_reference('p835_reference.csv')
        if not rows:
            pytest.skip("p835_reference.csv пуст")

        max_err_T = 0.0
        max_err_P = 0.0
        max_err_rho = 0.0

        for row in rows:
            z = float(row['z_km'])

            if 'T_k' in row and row['T_k']:
                T_ref = float(row['T_k'])
                T_my = global_temperature(z)
                err = abs(T_my - T_ref)
                max_err_T = max(max_err_T, err)

            if 'P_hpa' in row and row['P_hpa']:
                P_ref = float(row['P_hpa'])
                P_my = global_pressure(z)
                err = abs(P_my - P_ref) / max(P_ref, 1e-12)
                max_err_P = max(max_err_P, err)

            if 'rho_g_m3' in row and row['rho_g_m3']:
                rho_ref = float(row['rho_g_m3'])
                rho_my = global_water_vapour_density(z)
                err = abs(rho_my - rho_ref) / max(rho_ref, 1e-12) if rho_ref > 0 else abs(rho_my)
                max_err_rho = max(max_err_rho, err)

        # Пороги: T ±0.01 K, P ±0.1%, ρ ±5%
        assert max_err_T < 0.01, f"max_err_T = {max_err_T}"
        assert max_err_P < 0.001, f"max_err_P = {max_err_P}"
        assert max_err_rho < 0.05, f"max_err_rho = {max_err_rho}"
