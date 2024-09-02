import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np
import contextily as ctx
from scipy.stats import gaussian_kde
from tqdm import tqdm

# Устанавливаем имя города
place_name = "Yerevan, Armenia"

# Загружаем границы города
city_boundary = ox.geocode_to_gdf(place_name)

# Загружаем здания в пределах границ города
buildings = ox.features_from_place(place_name, tags={'building': True})

# Обрезаем здания по границам города
buildings_within_city = gpd.clip(buildings, city_boundary)

# Устанавливаем размер ячеек сетки 5 на 5 метров
grid_size = 0.00005  # Размер ячейки сетки в градусах (~5 метров)
minx, miny, maxx, maxy = city_boundary.total_bounds

# Создаем координаты с шагом grid_size с помощью numpy.arange
x_coords = np.arange(minx, maxx, grid_size)
y_coords = np.arange(miny, maxy, grid_size)

# Создаем пустой GeoDataFrame для сетки
grid_cells = []

# Добавляем tqdm для отображения прогресса
for x in tqdm(x_coords, desc="Создание сетки по X координатам"):
    for y in tqdm(y_coords, desc="Создание сетки по Y координатам", leave=False):
        grid_cells.append(box(x, y, x + grid_size, y + grid_size))

grid_cells_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=city_boundary.crs)

# Подсчитываем количество зданий в каждой ячейке с отображением прогресса
building_density = []
for cell in tqdm(grid_cells_gdf.geometry, desc="Подсчет плотности зданий"):
    density = buildings_within_city.within(cell).sum()
    building_density.append(density)

# Добавляем столбец с плотностью зданий
grid_cells_gdf['density'] = building_density

# Используем метод ядровой плотности для сглаживания данных
# Получаем координаты центров ячеек и плотность зданий
centroids = grid_cells_gdf.centroid
density = grid_cells_gdf['density']

# Создаем массив координат
coords = np.vstack([centroids.x, centroids.y])

# Применяем ядровую плотность с отображением прогресса
print("Вычисление ядровой плотности...")
kde = gaussian_kde(coords, weights=density, bw_method=0.05)
z = kde(coords)

# Добавляем ядровую плотность в GeoDataFrame
grid_cells_gdf['kde_density'] = z

# Визуализируем результат на карте с высоким разрешением
print("Создание карты...")
fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=200)  # Увеличенный размер и разрешение
city_boundary.to_crs(epsg=3857).plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
grid_cells_gdf.to_crs(epsg=3857).plot(column='kde_density', ax=ax, cmap='OrRd', legend=True, alpha=0.7)

# Добавляем подложку карты с высоким разрешением
ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, zoom=16)

ax.set_title('Detailed Building Density in Yerevan (5x5 meter grid)', fontsize=24)
ax.axis('off')
plt.show()
